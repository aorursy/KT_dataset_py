# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

%matplotlib inline

from xgboost import XGBRegressor
# Score function



def score_dataset(X_train, X_valid, y_train, y_valid):

    """Calculate the score for a specific approach.

    used model = RandomForestRegressor(n_estimators=100, random_state=0).



    OUTPUT: 

    mean_absolute_error: Total absolute error / n. Sum of total absolute error divided by number of samples.

    """

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



def get_score(n_estimators, X, y):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))

])

    

    from sklearn.model_selection import cross_val_score



    # Multiply by -1 since sklearn calculates *negative* MAE

    scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=3,

                              scoring='neg_mean_absolute_error')



    return scores.mean()



# Predict function



def predict (df1, df2, df3, estimators=100):

    """Predict for a specific approach.

    used model = RandomForestRegressor(n_estimators=100, random_state=0).



    INPUT:

    df1: training data

    df2: target

    df3: test data

    

    OUTPUT: 

    predictions for test data

    """

    # Define and fit model

    my_model = RandomForestRegressor(n_estimators=estimators, random_state=0)

    my_model.fit(df1, df2)



    # Get test predictions

    print ("Submission data have been calculated")

    return my_model.predict(df3)



# Save function



def save_file (predictions):

    """Save submission file."""

    # Save test predictions to file

    output = pd.DataFrame({'Id': sample_submission_file.Id,

                       'SalePrice': predictions})

    output.to_csv('submission.csv', index=False)

    print ("Submission file is saved")

    

def impute_numerical(df1, df2):

    """Impute to 2 dataframes that have only numerical values."""

    num_imputer = SimpleImputer(strategy='mean')



    num_X = pd.DataFrame(num_imputer.fit_transform(df1))

    num_test = pd.DataFrame(num_imputer.transform(df2))



    # Imputation removed column names; put them back

    num_X.columns = df1.columns

    num_test.columns = df2.columns

    return num_X, num_test





def impute_categorical(df1, df2):

    """Impute to 2 dataframes that have only categorical values."""

    

    cat_imputer = SimpleImputer(strategy='most_frequent')



    cat_X = pd.DataFrame(cat_imputer.fit_transform(df1))

    cat_test = pd.DataFrame(cat_imputer.transform(df2))



    # Imputation removed column names; put them back

    cat_X.columns = df1.columns

    cat_test.columns = df2.columns

    return cat_X, cat_test



print("Functions have been loaded!")
# For displaying all the columns of the data frame

pd.set_option('display.max_columns', None)



# get data

train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')



test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv", index_col='Id')



sample_submission_file = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")



with open('/kaggle/input/home-data-for-ml-course/data_description.txt', 'r') as f:

    description = f.read() 



# Create a dictionary of scores

scores_dict = {} # will be used for storing the scores of each approach.

submission_dict = {} # will be used for storing the submission scores of each approach.



print("Data have been loaded!")
train_data.head()
test_data.head(5)
sample_submission_file.head()
print ("Shape of train data: {}".format(train_data.shape))

print ("Shape of test data: {}".format(test_data.shape))

print ("Shape of submission file: {}".format(sample_submission_file.shape))
# Data descrition

print(description)
# Process Data



# Select target

y = train_data.SalePrice



# Just select numerical features!

X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

X_test = test_data.select_dtypes(exclude=['object']) # will be used for submision later...



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)

print ("Shapes:")

print ("X_train: {}".format(X_train.shape))

print ("X_valid: {}".format(X_valid.shape))

print ("y_train: {}".format(y_train.shape))

print ("y_valid: {}\n".format(y_valid.shape))



# Number of missing values in each column of training data

missing_val_count_by_column_train = (X_train.isnull().sum())



print('Missing value counts for numerical columns:')

print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



print ("Shapes:")

print ("reduced_X_train: {}".format(reduced_X_train.shape))

print ("reduced_X_valid: {}".format(reduced_X_valid.shape))
print("MAE from Approach 1 (Drop columns with missing values):")

scores_dict['A1'] = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A1'])
# Get names of columns with missing values of training data

cols_with_missing_X = [col for col in X.columns

                     if X[col].isnull().any()]

print ("Columns with missing values in training data: {}".format(cols_with_missing_X) + "\n")



# Get names of columns with missing values of test data

cols_with_missing_X_test = [col for col in X_test.columns

                     if X_test[col].isnull().any()]

print ("Columns with missing values in test data: {}".format(cols_with_missing_X_test) + "\n")



# Combine all missing columns

cols_with_missing = list(set(cols_with_missing_X).union(set(cols_with_missing_X_test)))



print ("Columns with missing values combined: {}".format(cols_with_missing))
# Drop missing values

X_a1 = X.drop(cols_with_missing, axis=1)

X_test_a1 = X_test.drop(cols_with_missing, axis=1)



print ("Shape of X_a1: {}".format(X_a1.shape))

print ("Shape of test_data_a1: {}".format(X_test_a1.shape))
# Get test predictions

preds_a1 = predict(X_a1, y, X_test_a1)



# Save test predictions to file

save_file(preds_a1)
submission_dict['A1'] = 17688.42490
# Process Data



# Select target

y = train_data.SalePrice



# Just select numerical features!

X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

X_test = test_data.select_dtypes(exclude=['object']) # will be used for submision later...



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)

print ("Shapes:")

print ("X_train: {}".format(X_train.shape))

print ("X_valid: {}".format(X_valid.shape))

print ("y_train: {}".format(y_train.shape))

print ("y_valid: {}\n".format(y_valid.shape))



# Number of missing values in each column of training data

missing_val_count_by_column_train = (X_train.isnull().sum())

print('Missing value counts for numerical columns:')

print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
# Imputation

imputed_X_train, imputed_X_valid = impute_numerical(X_train, X_valid)



print("MAE from Approach 2 (Imputation):")

scores_dict['A2'] = score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A2'])
# Imputation 

imputed_X, imputed_X_test = impute_numerical(X, X_test)



print ("Shape of imputed_X: {}".format(imputed_X.shape))

print ("Shape of imputed_test_data: {}".format(imputed_X_test.shape))
# Get test predictions

preds_a2 = predict(imputed_X, y, imputed_X_test)



# Save test predictions to file

save_file(preds_a2)
submission_dict['A2'] = 16546.14937
# Process Data



# Select target

y = train_data.SalePrice



# Just select numerical features!

X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

X_test = test_data.select_dtypes(exclude=['object']) # will be used for submision later...



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)

print ("Shapes:")

print ("X_train: {}".format(X_train.shape))

print ("X_valid: {}".format(X_valid.shape))

print ("y_train: {}".format(y_train.shape))

print ("y_valid: {}\n".format(y_valid.shape))



# Number of missing values in each column of training data

missing_val_count_by_column_train = (X_train.isnull().sum())

print('Missing value counts for numerical columns:')

print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()



# Imputation 

imputed_X_train_plus, imputed_X_valid_plus = impute_numerical(X_train_plus, X_valid_plus)



print("MAE from Approach 3 (An Extension to Imputation):")

scores_dict['A3'] = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A3'])
# Make copy to avoid changing original data (when imputing)

X_plus = X.copy()

reduced_X_test_plus = X_test.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_plus[col + '_was_missing'] = X_plus[col].isnull()

    reduced_X_test_plus[col + '_was_missing'] = reduced_X_test_plus[col].isnull()



# Imputation 

imputed_X_plus, imputed_reduced_X_test_plus = impute_numerical(X_plus, reduced_X_test_plus)



print ("Shape of imputed_X_plus: {}".format(imputed_X_plus.shape))

print ("Shape of imputed_reduced_X_test_plus: {}".format(imputed_reduced_X_test_plus.shape))
# Get test predictions

preds_a3 = predict(imputed_X_plus, y, imputed_reduced_X_test_plus)



# Save test predictions to file

save_file(preds_a3)
submission_dict['A3'] = 16423.75248
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)



# Get training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train: {}".format(X_train.shape))

print ("Shape of X_valid: {}".format(X_valid.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))



# Drop columns with missing values ()

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()] 

print()

print ("Columns with missing values in training data: {}".format(cols_with_missing) + "\n")



X_train.drop(cols_with_missing, axis=1, inplace=True)

X_valid.drop(cols_with_missing, axis=1, inplace=True)



print ("Shape of X_train after dropping missing values: {}".format(X_train.shape))

print ("Shape of X_valid after dropping missing values: {}".format(X_valid.shape))



# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print()

print("Categorical variables: {}\n".format(object_cols))



drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



print ("Shape of drop_X_train after dropping object values: {}".format(drop_X_train.shape))

print ("Shape of drop_X_valid after dropping object values: {}".format(drop_X_valid.shape))
print("MAE from Approach 4 (Drop categorical variables):")

scores_dict['A4'] = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A4'])
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()

print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}".format(y.shape))



# Drop columns with missing values ()

cols_with_missing_X = [col for col in X.columns if X[col].isnull().any()]

cols_with_missing_X_test = [col for col in X_test.columns if X_test[col].isnull().any()]

total_missing=set(cols_with_missing_X).union(set(cols_with_missing_X_test))

cols_with_missing=list(total_missing)



print()

print ("Columns with missing values in training data: {}".format(cols_with_missing) + "\n")



X.drop(cols_with_missing, axis=1, inplace=True)

X_test.drop(cols_with_missing, axis=1, inplace=True)



print ("Shape of X after dropping missing values: {}".format(X.shape))

print ("Shape of X_test after dropping missing values: {}".format(X_test.shape))



# Get list of categorical variables

s = (X.dtypes == 'object')

object_cols = list(s[s].index)



print()

print("Categorical variables: {}\n".format(object_cols))



drop_X = X.select_dtypes(exclude=['object'])

drop_X_test = X_test.select_dtypes(exclude=['object'])



print ("Shape of drop_X after dropping object values: {}".format(drop_X.shape))

print ("Shape of drop_X_test after dropping object values: {}".format(drop_X_test.shape))
# Get test predictions

preds_a4 = predict(drop_X, y, drop_X_test)



# Save test predictions to file

save_file(preds_a4)
submission_dict['A4'] = 17688.42490
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))



# Get training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train_full: {}".format(X_train_full.shape))

print ("Shape of X_valid_full: {}".format(X_valid_full.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}\n".format(y_valid.shape))



# Check Condition2 column

print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())



# All categorical columns

object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train_full[col]) == set(X_valid_full[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('\nCategorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



# Drop categorical columns that will not be encoded

label_X_train = X_train_full.drop(bad_label_cols, axis=1)

label_X_valid = X_valid_full.drop(bad_label_cols, axis=1)



print ("\nShape of label_X_train after dropping bad labels: {}".format(label_X_train.shape))

print ("Shape of label_X_valid after dropping bad labels: {}".format(label_X_valid.shape))
# Apply label encoder 



# https://stackoverflow.com/questions/46406720/labelencoder-typeerror-not-supported-between-instances-of-float-and-str

# Thanks to: @pceccon and @sgDysregulation

label_encoder = LabelEncoder()

for col in set(good_label_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train_full[col].astype(str))

    label_X_valid[col] = label_encoder.transform(X_valid_full[col].astype(str))



# Imputation to numerical columns

i_label_X_train, i_label_X_valid = impute_numerical(label_X_train, label_X_valid)



print("MAE from Approach 5 (Label Encoding):") 

scores_dict['A5'] = score_dataset(i_label_X_train, i_label_X_valid, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A5'])  
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))



# Find numerical columns

num_X = X.select_dtypes(exclude=['object']).copy()

num_test = test_data.select_dtypes(exclude=['object']).copy()



print ("Shape of num_X: {}".format(num_X.shape))

print ("Shape of num_test: {}".format(num_test.shape))



# All categorical columns

object_cols = X.columns



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X[col]) == set(X_test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('\nCategorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)



# Find categorical columns

cat_X = X[object_cols].copy()

cat_test = test_data[object_cols].copy()



print ("\nShape of cat_X: {}".format(cat_X.shape))

print ("Shape of cat_test: {}".format(cat_test.shape))



# Drop categorical columns that will not be encoded

cat_X = cat_X.drop(bad_label_cols, axis=1)

cat_test = cat_test.drop(bad_label_cols, axis=1)



print ("\nShape of cat_X after dropping bad label columns: {}".format(cat_X.shape))

print ("Shape of cat_test after dropping bad label columns: {}".format(cat_test.shape))
# Imputation and label encoding



# Imputation to numerical columns



i_num_X, i_num_test = impute_numerical(num_X, num_test)



print ("Shape of i_num_X: {}".format(i_num_X.shape))

print ("Shape of i_num_test: {}".format(i_num_test.shape))



# Imputation to categorical columns



i_cat_X, i_cat_test = impute_categorical(cat_X, cat_test)



print ("\nShape of i_cat_X: {}".format(i_cat_X.shape))

print ("Shape of i_cat_test: {}".format(i_cat_test.shape))



# Label encoding to categorical columns



label_encoder = LabelEncoder()

for col in set(good_label_cols):

    i_cat_X[col] = label_encoder.fit_transform(i_cat_X[col])

    i_cat_test[col] = label_encoder.transform(i_cat_test[col])



print ("\nShape of i_cat_X: {}".format(i_cat_X.shape))

print ("Shape of i_cat_test: {}".format(i_cat_test.shape))



# merge datasets



label_X = pd.concat([i_num_X, i_cat_X], axis=1)

label_test = pd.concat([i_num_test, i_cat_test], axis=1)



print ("\nShape of label_X after merge: {}".format(label_X.shape))

print ("Shape of label_test after merge: {}".format(label_test.shape))
# Get test predictions

preds_a5 = predict(label_X, y, label_test)



# Save test predictions to file

save_file(preds_a5)
submission_dict['A5'] = 16004.32251
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))



# Get training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train: {}".format(X_train.shape))

print ("Shape of X_valid: {}".format(X_valid.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))



object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('\nCategorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)



# Imputation to categorical columns



X_train_onehot, X_valid_onehot = impute_categorical(X_train[low_cardinality_cols], X_valid[low_cardinality_cols])



print ("\nShape of X_train_onehot: {}".format(X_train_onehot.shape))

print ("Shape of X_valid_onehot: {}".format(X_valid_onehot.shape))



# Apply one-hot encoder to each column with low_cardinality categorical data

OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_onehot))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_onehot))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train_onehot.index

OH_cols_valid.index = X_valid_onehot.index



# Remove categorical columns (will replace with one-hot encoding) and high_cardinality_cols

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



i_num_X_train, i_num_X_valid = impute_numerical(num_X_train, num_X_valid)



OH_X_train = pd.concat([i_num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([i_num_X_valid, OH_cols_valid], axis=1)



print ("\nShape of OH_X_train after merge: {}".format(OH_X_train.shape))

print ("Shape of OH_X_valid after merge: {}".format(OH_X_valid.shape))
print("MAE from Approach 6 (One-Hot Encoding):") 

scores_dict['A6'] = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid) # Store the score in the dictioanary

print(scores_dict['A6'])
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))



object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('\nCategorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)



# Imputation to categorical columns



X_onehot, test_data_onehot = impute_categorical(X[low_cardinality_cols], X_test[low_cardinality_cols])



print ("\nShape of X_onehot: {}".format(X_onehot.shape))

print ("Shape of test_data_onehot: {}".format(test_data_onehot.shape))



# Apply one-hot encoder to each column with low_cardinality categorical data

OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_onehot))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test_data_onehot))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_onehot.index

OH_cols_test.index = test_data_onehot.index



# Remove categorical columns (will replace with one-hot encoding) and high_cardinality_cols

num_X = X.drop(object_cols, axis=1)

num_test = test_data.drop(object_cols, axis=1)



i_num_X, i_num_test = impute_numerical(num_X, num_test)



OH_X_train = pd.concat([i_num_X, OH_cols_train], axis=1)

OH_X_test = pd.concat([i_num_test, OH_cols_test], axis=1)



print ("\nShape of OH_X_train: {}".format(OH_X_train.shape))

print ("Shape of OH_X_test: {}".format(OH_X_test.shape))
# Get test predictions

preds_a6 = predict(OH_X_train, y, OH_X_test)



# Save test predictions to file

save_file(preds_a6)
submission_dict['A6'] = 16073.17499
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))





# Get training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train_full: {}".format(X_train_full.shape))

print ("Shape of X_valid_full: {}".format(X_valid_full.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()



print ("\nShape of X_train: {}".format(X_train.shape))

print ("Shape of X_valid: {}".format(X_valid.shape))
# Define transformers

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
# Define the Model

model = RandomForestRegressor(n_estimators=100, random_state=0)
# Create and Evaluate the Pipeline

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

scores_dict['A7'] = mean_absolute_error(y_valid, preds) # Store the score in the dictioanary



print('MAE from Approach 7 (pipelines):')

print(scores_dict['A7'])

# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))





# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols



# Preprocessing of training data 

pipeline_X = X[my_cols].copy()

pipeline_X_test = X_test[my_cols].copy()



print ("\nShape of X: {}".format(pipeline_X.shape))

print ("Shape of X_test: {}\n".format(pipeline_X_test.shape))



# Fit model

my_pipeline.fit(pipeline_X, y)



# Get predictions

preds_a7 = my_pipeline.predict(pipeline_X_test)



# Save test predictions to file

save_file(preds_a7)

submission_dict['A7'] = 16073.17499
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))





# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]





cv_X = X[numerical_cols].copy()

cv_test = X_test[numerical_cols].copy()



print ("\nShape of cv_X: {}".format(cv_X.shape))

print ("Shape of cv_test: {}".format(cv_test.shape))
results = {}

my_list = [50, 100, 150, 200, 250, 300, 350, 400]

for num in my_list:

    results.update({num : get_score(num,cv_X, y)})

    print (results.get(num))
scores_dict['A8']=min(results.values()) # Store the score in the dictioanary



# https://stackoverflow.com/questions/43431347/python-dictionary-plot-matplotlib/43431522

# Thanks to: @tanaka, @LucG

fig = plt.figure(figsize=[20, 10])

plt.plot(list(results.keys()),list(results.values()))
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))



categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols
# Define transformers

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
# Define the Model

model = RandomForestRegressor(n_estimators=200, random_state=0)
# Preprocessing of training data, fit model 

cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

cv_X = X[my_cols].copy()

cv_X_test = X_test[my_cols].copy()

cv_pipeline.fit(cv_X, y)



# Preprocessing of validation data, get predictions

preds_a8 = cv_pipeline.predict(cv_X_test)



# Save test predictions to file

save_file(preds_a8)
submission_dict['A8'] = 15950.53953
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)





print ("Shape of X: {}".format(X.shape))

print ("Shape of y: {}\n".format(y.shape))





# Get training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train_full: {}".format(X_train_full.shape))

print ("Shape of X_valid_full: {}".format(X_valid_full.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))





# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols



X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()





# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)



print ("\nShape of X_train after encoding: {}".format(X_train.shape))

print ("Shape of y_train after encoding: {}".format(y_train.shape))

print ("Shape of X_valid after encoding: {}".format(X_valid.shape))

print ("Shape of y_valid after encoding: {}".format(y_valid.shape))
# Train model

my_model = XGBRegressor(random_state=0)

my_model.fit(X_train, y_train)

# Predict

prediction_1 = my_model.predict(X_valid)



# Calculate MAE

scores_dict['A9'] = mean_absolute_error(prediction_1, y_valid) # Store the score in the dictioanary



# print MAE

print("Mean Absolute Error:\n" , scores_dict['A9'])
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test = test_data.copy()



print ("Shape of X: {}".format(X.shape))

print ("Shape of X_test: {}".format(X_test.shape))

print ("Shape of y: {}\n".format(y.shape))





# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X = X[my_cols]

X_test = X_test[my_cols]



# One-hot encode the data (to shorten the code, we use pandas)

X = pd.get_dummies(X)

X_test = pd.get_dummies(X_test)

X, X_test = X.align(X_test, join='left', axis=1)



print ("\nShape of X after encoding: {}".format(X.shape))

print ("Shape of X_test after encodin: {}".format(X_test.shape))





my_model = XGBRegressor(random_state=0)

my_model.fit(X, y)

preds_a9 = my_model.predict(X_test)



# Save test predictions to file

save_file(preds_a9)
submission_dict['A9'] = 16011.84256
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)





print ("Shape of X: {}".format(X.shape))

print ("Shape of y: {}\n".format(y.shape))





# Get training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train_full: {}".format(X_train_full.shape))

print ("Shape of X_valid_full: {}".format(X_valid_full.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))





# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols



X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()





# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)



print ("\nShape of X_train after encoding: {}".format(X_train.shape))

print ("Shape of y_train after encoding: {}".format(y_train.shape))

print ("Shape of X_valid after encoding: {}".format(X_valid.shape))

print ("Shape of y_valid after encoding: {}".format(y_valid.shape))

# Define the model

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=2)



# Fit the model

my_model_2.fit(X_train, y_train,

               early_stopping_rounds=5,

               eval_set=[(X_valid, y_valid)],

               verbose=False)

               



# Get predictions

predictions_2 = my_model_2.predict(X_valid)



# Calculate MAE

scores_dict['A10'] = mean_absolute_error(predictions_2, y_valid) # Store the score in the dictioanary



# Uncomment to print MAE

print("Mean Absolute Error:\n" , scores_dict['A10'])
# Prepare data



# Separate target from predictors

y = train_data.SalePrice

X = train_data.drop(['SalePrice'], axis=1)

X_test_full = test_data.copy()





print ("Shape of X: {}".format(X.shape))

print ("Shape of y: {}".format(y.shape))

print ("Shape of X_test_full: {}\n".format(X_test_full.shape))



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print ("Shape of X_train_full: {}".format(X_train_full.shape))

print ("Shape of X_valid_full: {}".format(X_valid_full.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of y_valid: {}".format(y_valid.shape))



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()





# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)



print ("\nShape of X_train after encoding: {}".format(X_train.shape))

print ("Shape of y_train after encoding: {}".format(y_train.shape))

print ("Shape of X_valid after encoding: {}".format(X_valid.shape))

print ("Shape of y_valid after encoding: {}".format(y_valid.shape))

print ("Shape of X_test after encoding: {}".format(X_test.shape))
# Define the model

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=2)



# Fit the model

my_model_2.fit(X_train, y_train,

               early_stopping_rounds=5,

               eval_set=[(X_valid, y_valid)],

               verbose=False)

               



prediction_2 = my_model_2.predict(X_test)



# Save test predictions to file

save_file(prediction_2)

submission_dict['A10'] = 14810.12828
print(scores_dict)

print(submission_dict)
# Plot Validation Score ve Actual Results figure

Y1=[x for x in scores_dict.values()]

Y2=[x for x in submission_dict.values()]



X1=[x for x in scores_dict.keys()]

X2=[x for x in submission_dict.keys()]



fig = plt.figure(figsize=[20, 10])

plt.xlabel('Approaches')

plt.ylabel('Score')

plt.title('Validation ve Actual Scores')

plt.plot(X1,Y1, color='tab:blue')

plt.plot(X2,Y2, color='tab:orange')



plt.legend(["Validation Score", "Actual Result"])