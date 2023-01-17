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



#Load training and validation from csv

X_full_exploratory = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")  #temp, to get access to target...

X_full = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

X_test_full = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")



#drop the rows in the target that have missing values (ie: we cant use them for training or validating)

X_full.dropna(axis=0,subset=['SalePrice'], inplace=True)

y = X_full.SalePrice



#drop SalePrice from X_full as it is the target

X_full.drop('SalePrice', axis=1, inplace=True)



#break off validation set from training data 

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,  train_size=0.8, test_size = 0.2, random_state=1)







#Select Categorical values with low cardinality ("Cardinality" means the number of unique values in a column)

categorical_columns = [cname for cname in X_full.columns if 

                       X_full[cname].nunique() < 10 and

                       X_full[cname].dtype == 'object']



# Select numerical columns

numerical_colums = [cname for cname in X_full.columns if

                   X_full[cname].dtype in ['int64', 'float64']]



#Keeping selected columns only

my_columns = categorical_columns + numerical_colums

X_train = X_train_full[my_columns].copy()

X_valid = X_valid_full[my_columns].copy()

X_test = X_test_full[my_columns].copy()





#dropped columns

dropped_columns = [cname for cname in X_full.columns if cname not in my_columns]

X_full[dropped_columns]

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder



#numerical columns

numerical_columns_list = [colname for colname in X_train.columns if

                    X_train[colname].dtypes in ['int64', 'float64']]



#categorical columns

categorical_columns_list = [colname for colname in X_train.columns if

                    X_train[colname].dtypes == 'object' ]





X_train_trf = X_train.copy()

X_valid_trf = X_valid.copy()

X_test_trf = X_test.copy()



# Preprocessing for numerical data

num_imputer = SimpleImputer(strategy='median')

X_train_trf[numerical_columns_list] = num_imputer.fit_transform(X_train_trf[numerical_columns_list])

X_valid_trf[numerical_columns_list] = num_imputer.transform(X_valid_trf[numerical_columns_list])

X_test_trf[numerical_columns_list] = num_imputer.transform(X_test_trf[numerical_columns_list])



# Preprocessing for categorical data

cat_imputer = SimpleImputer(strategy='most_frequent')

X_train_trf[categorical_columns_list] = cat_imputer.fit_transform(X_train_trf[categorical_columns_list])

X_valid_trf[categorical_columns_list] = cat_imputer.transform(X_valid_trf[categorical_columns_list])

X_test_trf[categorical_columns_list] = cat_imputer.transform(X_test_trf[categorical_columns_list])



le = LabelEncoder()

for col in X_train_trf[categorical_columns_list].columns:

    X_train_trf[col] = le.fit_transform(X_train_trf[col])

    X_valid_trf[col] = le.fit_transform(X_valid_trf[col])

    X_test_trf[col] = le.fit_transform(X_test_trf[col])

X_valid_trf
X_test_trf
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error







#create the model 

model = XGBRegressor(n_estimators=1000, learning_rate=0.05,

                     subsample=0.8, colsample_bytree= 0.8, seed=42)



model.fit(X_train_trf,y_train,

        early_stopping_rounds=5,

        eval_set=[(X_train_trf, y_train), (X_valid_trf, y_valid)],

        verbose=False)

preds = model.predict(X_valid_trf)

score = mean_absolute_error(preds, y_valid)

print("MAE:", score)



preds
X_valid_trf
#Id should be int32

preds = model.predict(X_test_trf)

X_test_trf.Id = X_test_trf.Id.astype('int32')



output = pd.DataFrame({'Id': X_test_trf.Id,

                       'SalePrice': preds})



output.to_csv('submission.csv', index=False)
output