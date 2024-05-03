import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Read data
stroke_df = pd.read_csv('data/processed_data/processed_stroke.csv')
heart_df = pd.read_csv('data/processed_data/processed_heart.csv')

# Binarize age and Age in each dataset
stroke_median = stroke_df['age'].median()
stroke_df['age'] = stroke_df['age'].apply(lambda x: 1 if x > stroke_median else 0)

heart_median = heart_df['Age'].median()
heart_df['Age'] = heart_df['Age'].apply(lambda x: 1 if x < heart_median else 0)

# Split data into train and test sets (70% train, 30% test)
# case 1: base case
stroke_train, stroke_test = train_test_split(stroke_df, test_size=0.3, random_state=42)
heart_train, heart_test = train_test_split(heart_df, test_size=0.3, random_state=42)

# Define functions
def remove_protected_classes(a, df):
    return df.drop(columns=a)

def flip_values(x):
    return 1 - x

# case 2: Remove protected classes
stroke_no_protected_train = remove_protected_classes(['gender', 'age'], stroke_train)
stroke_no_protected_test = remove_protected_classes(['gender', 'age'], stroke_test)

heart_no_protected_train = remove_protected_classes(['Age', 'Sex'], heart_train)
heart_no_protected_test = remove_protected_classes(['Age', 'Sex'], heart_test)

# case 3: Flip values
stroke_dup_train = stroke_train.copy()
stroke_dup_train['age'] = stroke_dup_train['age'].apply(lambda x: flip_values(x))
stroke_dup_train['gender'] = stroke_dup_train['gender'].apply(lambda x: flip_values(x))
stroke_augmented_data_train = pd.concat([stroke_train, stroke_dup_train])

heart_dup_train = heart_train.copy()
heart_dup_train['Age'] = heart_dup_train['Age'].apply(lambda x: flip_values(x))
heart_dup_train['Sex'] = heart_dup_train['Sex'].apply(lambda x: flip_values(x))
heart_augmented_data_train = pd.concat([heart_train, heart_dup_train])

# case 4: Use SMOTE
smote = SMOTE(random_state=42)

X_train, y_train = stroke_train.drop('stroke', axis=1), stroke_train['stroke']
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
smote_stroke_train = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.DataFrame(y_resampled, columns=['stroke'])], axis=1)

X_train, y_train = heart_train.drop('HeartDisease', axis=1), heart_train['HeartDisease']
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
smote_heart_train = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.DataFrame(y_resampled, columns=['HeartDisease'])], axis=1)

# case 5: SMOTE remove protected classes
smote_no_prot_class_stroke_train = remove_protected_classes(['gender', 'age'], smote_stroke_train)
smote_no_prot_class_stroke_test = remove_protected_classes(['gender', 'age'], stroke_test)

smote_no_prot_class_heart_train = remove_protected_classes(['Age', 'Sex'], smote_heart_train)
smote_no_prot_class_heart_test = remove_protected_classes(['Age', 'Sex'], heart_test)

# case 6: SMOTE augment the dataset
smote_stroke_dup_train = smote_stroke_train.copy()
smote_stroke_dup_train['age'] = smote_stroke_dup_train['age'].apply(lambda x: flip_values(x))
smote_stroke_dup_train['gender'] = smote_stroke_dup_train['gender'].apply(lambda x: flip_values(x))
smote_stroke_augmented_data_train = pd.concat([smote_stroke_train, smote_stroke_dup_train])

smote_heart_dup_train = heart_train.copy()
smote_heart_dup_train['Age'] = smote_heart_dup_train['Age'].apply(lambda x: flip_values(x))
smote_heart_dup_train['Sex'] = smote_heart_dup_train['Sex'].apply(lambda x: flip_values(x))
smote_heart_augmented_data_train = pd.concat([smote_heart_train, smote_heart_dup_train])

# Save data
# case 1
base_case_stroke_train = stroke_train
base_case_stroke_test = stroke_test
base_case_stroke_train.to_csv('data/train_test_data/base_case_stroke_train.csv', index=False)
base_case_stroke_test.to_csv('data/train_test_data/base_case_stroke_test.csv', index=False)

base_case_heart_train = heart_train
base_case_heart_test = heart_test
base_case_heart_train.to_csv('data/train_test_data/base_case_heart_train.csv', index=False)
base_case_heart_test.to_csv('data/train_test_data/base_case_heart_test.csv', index=False)

# case 2
stroke_no_protected_train.to_csv('data/train_test_data/no_prot_class_stroke_train.csv', index=False)
stroke_no_protected_test.to_csv('data/train_test_data/no_prot_class_stroke_test.csv', index=False)

heart_no_protected_train.to_csv('data/train_test_data/no_prot_class_heart_train.csv', index=False)
heart_no_protected_test.to_csv('data/train_test_data/no_prot_class_heart_test.csv', index=False)

# case 3
stroke_augmented_data_train.to_csv('data/train_test_data/augmented_stroke_train.csv', index=False)
stroke_augmented_data_test = stroke_test
stroke_augmented_data_test.to_csv('data/train_test_data/augmented_stroke_test.csv', index=False)

heart_augmented_data_train.to_csv('data/train_test_data/augmented_heart_train.csv', index=False)
heart_augmented_data_test = heart_test
heart_augmented_data_test.to_csv('data/train_test_data/augmented_heart_test.csv', index=False)

# case 4
smote_stroke_train.to_csv('data/train_test_data/smote_stroke_train.csv', index=False)
smote_stroke_test = stroke_test
smote_stroke_test.to_csv('data/train_test_data/smote_stroke_test.csv', index=False)

smote_heart_train.to_csv('data/train_test_data/smote_heart_train.csv', index=False)
smote_heart_test = heart_test
smote_heart_test.to_csv('data/train_test_data/smote_heart_test.csv', index=False)

# case 5
smote_no_prot_class_stroke_train.to_csv('data/train_test_data/smote_no_prot_class_stroke_train.csv', index=False)
smote_no_prot_class_stroke_test.to_csv('data/train_test_data/smote_no_prot_class_stroke_test.csv', index=False)

smote_no_prot_class_heart_train.to_csv('data/train_test_data/smote_no_prot_class_heart_train.csv', index=False)
smote_no_prot_class_heart_test.to_csv('data/train_test_data/smote_no_prot_class_heart_test.csv', index=False)

# case 6
smote_stroke_augmented_data_train.to_csv('data/train_test_data/smote_stroke_augmented_data_train.csv', index=False)
smote_stroke_augmented_data_test = stroke_test
smote_stroke_augmented_data_test.to_csv('data/train_test_data/smote_stroke_augmented_data_test.csv', index=False)

smote_heart_augmented_data_train.to_csv('data/train_test_data/smote_heart_augmented_data_train.csv', index=False)
smote_heart_augmented_data_test = heart_test
smote_heart_augmented_data_test.to_csv('data/train_test_data/smote_heart_augmented_data_test.csv', index=False)
