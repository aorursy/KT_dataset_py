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
train = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

train.shape, test.shape 
#drop rows where target has NA values

X = train.dropna(axis=0, subset=['SalePrice'])

y = train.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)

X.shape, y.shape
categorical_variables = X.select_dtypes(include='object').columns.tolist()

numerical_variables = X.select_dtypes(exclude='object').columns.tolist()

print(categorical_variables, numerical_variables)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=200, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)

#reduce columns with missing values

s = X_train.isnull().sum()

reduce_cols = s[s > 0].index.to_list()



reduced_X_train = X_train.drop(reduce_cols + categorical_variables, axis=1)

reduced_X_valid = X_valid.drop(reduce_cols + categorical_variables, axis=1)



print("MAE from Approach 1 (drop na cols):\n", score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)) 

from sklearn.impute import SimpleImputer



sin = SimpleImputer(strategy= 'median')

X_train_n = pd.DataFrame(sin.fit_transform(X_train[numerical_variables]))

X_train_n.columns = numerical_variables

X_valid_n = pd.DataFrame(sin.transform(X_valid[numerical_variables]))

X_valid_n.columns = numerical_variables





sic = SimpleImputer(strategy= 'most_frequent')

X_train_c = pd.DataFrame(sic.fit_transform(X_train[categorical_variables]))

X_train_c.columns = categorical_variables

X_valid_c = pd.DataFrame(sic.transform(X_valid[categorical_variables]))

X_valid_c.columns = categorical_variables



imputed_X_train = pd.concat([X_train_n, X_train_c], axis=1)

imputed_X_valid = pd.concat([X_valid_n, X_valid_c], axis=1)



#only numerical var

print("MAE from Approach 2 (Imputation):") 

print(score_dataset(imputed_X_train[numerical_variables], imputed_X_valid[numerical_variables], y_train, y_valid))



#both

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

low_cardinality_cols = [col for col in categorical_variables if imputed_X_train[col].nunique()<10]

high_cardinality_cols = [col for col in categorical_variables if imputed_X_valid[col].nunique()>=10]

#print(len(low_cardinality_cols), len(high_cardinality_cols))



ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

oh_cols_train = pd.DataFrame(ohe.fit_transform(imputed_X_train[low_cardinality_cols])) 

oh_cols_valid = pd.DataFrame(ohe.transform(imputed_X_valid[low_cardinality_cols])) 



oh_cols_train.index = imputed_X_train.index

oh_cols_valid.index = imputed_X_valid.index



num_X_train = imputed_X_train.drop(low_cardinality_cols + high_cardinality_cols, axis=1)

num_X_valid = imputed_X_valid.drop(low_cardinality_cols + high_cardinality_cols, axis=1)



OH_X_train = pd.concat([num_X_train, oh_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, oh_cols_valid], axis=1)



print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
df1 = pd.concat([OH_X_train, OH_X_valid], axis=0)

df1.reset_index(drop=True, inplace=True)

df2 = pd.concat([y_train, y_valid], axis=0)

df2.columns=['SalePrice']

df2.reset_index(drop=True, inplace=True)

df = pd.concat([df1, df2], axis=1)

c = abs(df.corr()['SalePrice']).sort_values(ascending=False)

high_corr = c[c>0.01][1:].index.tolist()

len(high_corr)
from xgboost import XGBRegressor

def score_dataset_x(X_train, X_valid, y_train, y_valid):

    model = XGBRegressor(n_estimators=1100, learning_rate=0.01, random_state=0, n_jobs=3)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)



print("MAE from Approach 4 (XGB):") 

print(score_dataset_x(OH_X_train[high_corr], OH_X_valid[high_corr], y_train, y_valid))
model = XGBRegressor(n_estimators=1100, learning_rate=0.01, random_state=0, n_jobs=3)

model.fit(pd.concat([OH_X_train[high_corr], OH_X_valid[high_corr]], axis=0), pd.concat([y_train, y_valid], axis=0))



#on test

X_test_n = pd.DataFrame(sin.transform(test[numerical_variables]))

X_test_n.columns = numerical_variables

X_test_c = pd.DataFrame(sic.transform(test[categorical_variables]))

X_test_c.columns = categorical_variables

imputed_X_test = pd.concat([X_test_n, X_test_c], axis=1)

oh_cols_test = pd.DataFrame(ohe.transform(imputed_X_test[low_cardinality_cols])) 

oh_cols_test.index = imputed_X_test.index

num_X_test = imputed_X_test.drop(low_cardinality_cols + high_cardinality_cols, axis=1)

OH_X_test = pd.concat([num_X_test, oh_cols_test], axis=1) 

preds_test = model.predict(OH_X_test[high_corr])

output = pd.DataFrame({'Id': test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

print('Done!')
c[c>0.08]