# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', 200)
### Load Train and Test Data



prefix = '/kaggle/input/home-data-for-ml-course'



X = pd.read_csv(prefix + '/train.csv', index_col='Id') 

X_test = pd.read_csv(prefix + '/test.csv', index_col='Id')



X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice



X.drop(['SalePrice'], axis=1, inplace=True)

test_ids = X_test.index
X.info()
X.describe()
X.head()
X_test.head()
### Handle Missing Values



cols_with_missing = list(

  set([col for col in X.columns if X[col].isnull().any()]) |

  set([col for col in X_test.columns if X_test[col].isnull().any()]))

print('\nColumns with missing values:\n', cols_with_missing)



missing_val_count_by_column = X.isnull().sum()

print('\nMissing values per column:\n', missing_val_count_by_column[missing_val_count_by_column > 0])



total_rows_count = X.shape[0] + X_test.shape[0]

threshold = 0.9 # 90%

cols_missing_drop = (missing_val_count_by_column[

    missing_val_count_by_column > total_rows_count * (1 - threshold)]).index.values

cols_missing_impute = list(set(cols_with_missing) - set(cols_missing_drop))

print('\nColumns to be dropped:\n', cols_missing_drop)

print('\nColumns to be imputed:\n', cols_missing_impute)
X[cols_missing_drop].head()
X[cols_missing_impute].head()
X.drop(cols_missing_drop, axis=1, inplace=True)

X_test.drop(cols_missing_drop, axis=1, inplace=True)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='most_frequent')

imputed_X = pd.DataFrame(imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(imputer.transform(X_test))



imputed_X.columns = X.columns

imputed_X_test.columns = X_test.columns
#missing_val_count_by_column = (imputed_X_test.isnull().sum())

#print('\nNumber of missing values per column:\n', missing_val_count_by_column[missing_val_count_by_column > 0])



X = imputed_X

X_test = imputed_X_test
### Handle Categorical Variables



object_cols = [col for col in X.columns if X[col].dtype == "object"]

good_label_cols = [col for col in object_cols if set(X[col]) == set(X_test[col])]

bad_label_cols = list(set(object_cols) - set(good_label_cols))



print('\nAll categorical columns in the dataset:\n', object_cols)

print('\nCategorical columns that will be label encoded:\n', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:\n', bad_label_cols)
X[bad_label_cols].head()
X[good_label_cols].head()
X[good_label_cols].info()
from sklearn.preprocessing import LabelEncoder



label_X = X.drop(bad_label_cols, axis=1)

label_X_test = X_test.drop(bad_label_cols, axis=1)



label_encoder = LabelEncoder()

for col in good_label_cols:

    label_X[col] = label_encoder.fit_transform(X[col])

    label_X_test[col] = label_encoder.transform(X_test[col])
X = label_X

X_test = label_X_test



object_cols = good_label_cols
X.head()
object_nunique = list(map(lambda col: X[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



nunique = sorted(d.items(), key=lambda x: x[1])

print('Number of unique entries in each column with categorical data:', nunique)
low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]

high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

#high_cardinality_cols = [k for k, v in d.items() if v > 10]



print('\nCategorical columns that will be one-hot encoded:\n', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:\n', high_cardinality_cols)
X[low_cardinality_cols].head()
X[high_cardinality_cols].head()
from sklearn.preprocessing import OneHotEncoder



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))



OH_cols.index = X.index

OH_cols_test.index = X_test.index



num_X = X.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)



OH_X = pd.concat([num_X, OH_cols], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
X = OH_X

X_test = OH_X_test
X.head()
X_test.head()
X.columns
final_X = X

final_y = y

final_X_test = X_test
#model = RandomForestRegressor(n_estimators=200, random_state=0)

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)



model.fit(final_X, final_y)

preds_test = model.predict(final_X_test)
output = pd.DataFrame({'Id': test_ids,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
output.head()