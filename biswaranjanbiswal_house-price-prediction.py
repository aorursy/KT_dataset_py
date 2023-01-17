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
training=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
training.shape
test.shape
training['train_test']=1

test['train_test']=0

combined=pd.concat([training,test])

combined.shape
combined.head(5)
training.head(10)
test.head(10)
#take care of missing data

combined.isnull().sum().sort_values(ascending=False)[:36]
sns.heatmap(combined.isnull(), cbar= False, cmap = 'magma')
# Storing the columns with missing values in a list

Missing_val_col = [col for col in combined.columns if combined[col].isnull().sum() > 1]

Missing_val_col
# Checking the percentage of missing values. If any feature has more than 40% missing value then drop that feature. 

for col in Missing_val_col:

    if (combined[col].isnull().sum() > int(0.40 * combined.shape[0])):

        print(col) 
#drop these columns

combined.drop(['Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature','Id'],axis=1, inplace = True)
combined.shape
#Checking the numerical features. 

numerical_data = [col for col in combined.columns if combined[col].dtypes != 'O']

combined[numerical_data].head(5)

# Missing value features in numerical type column

numerical_missing = [col for col in numerical_data if combined[col].isnull().sum() > 1 ]

numerical_missing

# Checking categorical features and treating missing values in that

category_data = [col for col in combined.columns if combined[col].dtypes == 'O']

combined[category_data].head(5)
category_missing = [col for col in category_data if combined[col].isnull().sum() > 1 ]

category_missing
#let's fill the data

combined['LotFrontage'].fillna(combined['LotFrontage'].mean(), inplace = True)

combined['MasVnrArea'].fillna(combined['MasVnrArea'].mode()[0], inplace = True)

combined['GarageYrBlt'].fillna(combined['GarageYrBlt'].mode()[0], inplace = True)

combined['MSZoning'].fillna(combined['MSZoning'].mode()[0], inplace = True)

combined['Utilities'].fillna(combined['Utilities'].mode()[0], inplace = True)

combined['BsmtFullBath'].fillna(combined['BsmtFullBath'].mode()[0], inplace = True)

combined['BsmtHalfBath'].fillna(combined['BsmtHalfBath'].mode()[0], inplace = True)

combined['Functional'].fillna(combined['Functional'].mode()[0], inplace = True)

combined['Electrical'].fillna(combined['Functional'].mode()[0], inplace = True)







combined['TotalBsmtSF'].fillna(combined['TotalBsmtSF'].mode()[0], inplace = True)

combined['GarageArea'].fillna(combined['GarageArea'].mode()[0], inplace = True)

combined['BsmtUnfSF'].fillna(combined['BsmtUnfSF'].mode()[0], inplace = True)

combined['SaleType'].fillna(combined['SaleType'].mode()[0], inplace = True)

combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0], inplace = True)

combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0], inplace = True)

combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0], inplace = True)

combined['BsmtFinSF1'].fillna(combined['BsmtFinSF1'].mode()[0], inplace = True)

combined['GarageCars'].fillna(combined['GarageCars'].mode()[0], inplace = True)

combined['BsmtFinSF2'].fillna(combined['BsmtFinSF2'].mode()[0], inplace = True)



# Replacing categorical data with mode

for i in category_missing:

    combined[i].fillna(combined[i].mode()[0], inplace = True)

sns.heatmap(combined.isnull(), cbar= False, cmap = 'magma')
combined.isnull().sum().sort_values(ascending=False)[:2]
df = pd.get_dummies(combined[category_data],

                           columns = category_data,drop_first=True)

df
combined.shape
combined=pd.concat([combined,df],axis=1)
combined.shape
combined=combined.drop(category_data,axis=1)
combined.shape
X_train = combined[combined.train_test == 1].drop(['train_test'], axis =1)

X_test = combined[combined.train_test == 0].drop(['train_test'], axis =1)

y_train = combined[combined.train_test==1].SalePrice
import xgboost

classifier=xgboost.XGBRegressor()

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

basic_output = {'Id': test.Id, 'SalePrice': y_pred}

basic_output=pd.DataFrame(data=basic_output)

basic_output.to_csv('basic_output.csv', index=False)