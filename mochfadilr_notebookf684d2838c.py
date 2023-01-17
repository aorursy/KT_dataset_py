# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

# data_descript = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt')
y = train['SalePrice']

X = train.drop(['Id', 'SalePrice'], axis=1)
X_train.info()
#Take the list of columns with more than 300 missing value

cols_with_missing = [col for col in X

                     if X[col].isnull().sum() > 300]



X[cols_with_missing].describe()
for i in cols_with_missing:

    X[i] = X[i].fillna("NA")
X[cols_with_missing].info()
#Visualize the dataset to check are there lots of outliers in it

X.hist(bins = 50, figsize = (20, 15))

plt.show()
#Data normalization with feature clipping technique to deal with outliers

X['1stFlrSF'] = np.clip(X['1stFlrSF'], a_min = None, a_max = 2250) 

X['2ndFlrSF'] = np.clip(X['2ndFlrSF'], a_min = None, a_max = 1250)

X['LotArea'] = np.clip(X['LotArea'], a_min = None, a_max = 50000) 

X['GarageArea'] = np.clip(X['GarageArea'], a_min = None, a_max = 1000) 

X['BsmtFinSF1'] = np.clip(X['BsmtFinSF1'], a_min = None, a_max = 2000) 

X['GrLivArea'] = np.clip(X['GrLivArea'], a_min = 500, a_max = 3250)

X['LotFrontage'] = np.clip(X['LotFrontage'], a_min = None, a_max = 150) 

X['MasVnrArea'] = np.clip(X['MasVnrArea'], a_min = None, a_max = 500) 

X['TotalBsmtSF'] = np.clip(X['TotalBsmtSF'], a_min = None, a_max = 2500)

X['OpenPorchSF'] = np.clip(X['OpenPorchSF'], a_min = None, a_max = 175)
X.hist(bins = 50, figsize = (20, 15))

plt.show()
#create categorical columns

obj_cols = [cols for cols in X.columns if X[cols].dtype == 'object']

#create numerical columns

num_cols = [cols for cols in X.columns if X[cols].dtype in ['int64', 'float64']]



#Impute missing values in numerical columns with median value

num_transform = Pipeline(steps=[('scaler', StandardScaler()),

                               ('imputer', SimpleImputer(strategy='median'))])



#Impute missing values in categorical columns with most frequent value

#Change the categorical column to numerical column with One-hot encoder

obj_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                              ('OH_encode', OneHotEncoder(handle_unknown='ignore'))])



#Applies transformers to columns in the dataset

preprocessor = ColumnTransformer(transformers=[('num', num_transform, num_cols),

                                          ('obj', obj_transform, obj_cols)])



#modeling

model = XGBRegressor(learning_rate=0.03,

                     n_estimators=600,

                     random_state=1)



my_model = Pipeline(steps=[('preprocessor', preprocessor),

                        ('model', model)])



my_model.fit(X, y)
predict = my_model.predict(test)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict

    })

submission.to_csv('submission.csv', index=False)