import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from scipy import stats

from scipy.stats import norm, skew



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score, GridSearchCV
#Reading datasets train and test

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Taking a copy

train_copy = df_train.copy()

test_copy = df_test.copy()
df_train.head()
df_test.head()
# Review the dataset shape

df_train.shape, df_test.shape
#Drop ID column in train and test dataset as it has no influence over target

df_train.drop('Id', inplace=True, axis=1)

df_test.drop('Id', inplace=True, axis=1)
#Review dataset shape

df_train.shape, df_test.shape
# Handling Outliers

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.xlabel('Ground Live Area')

plt.ylabel('SalePrice')

plt.title('Sale Price VS Living Area')

plt.show()
#Let's remove ground area >4000

df_train[df_train['GrLivArea']>4000][['GrLivArea','SalePrice']]
df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, axis=0)
# Review data for Outliers

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])

plt.xlabel('Ground Live Area')

plt.ylabel('SalePrice')

plt.title('Sale Price VS Living Area')

plt.show()
#Get total number of records for each dataset

ntrain = df_train.shape[0]

ntest = df_test.shape[0]

ntrain, ntest
#Get target value

y_train = df_train['SalePrice']
df_alldata = pd.concat([df_train, df_test]).reset_index(drop=True)

df_alldata.head()
#Drop Target column from allData

df_alldata.drop('SalePrice', inplace=True, axis=1)
df_alldata.shape
#Find missing columns with missing data

alldata_na = df_alldata.isnull().sum()

alldata_na = alldata_na[alldata_na>0]

alldata_na = alldata_na.sort_values(ascending=False)

print('Number of Cols with Null values to handle: ',len(alldata_na))
#For selected columns below impute missing values with 'None'

nonecols=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType','MSSubClass']



for col in nonecols:

    df_alldata[col] = df_alldata[col].fillna('None')
#For selected columns below impute missing values with 0

zerocols = ['GarageYrBlt','GarageArea','GarageCars', 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']



for col in zerocols:

    df_alldata[col]=df_alldata[col].fillna(0)
#For selected columns below fill null with mode

modecols=['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']



for col in modecols:

    df_alldata[col] = df_alldata[col].fillna(df_alldata[col].mode()[0])
#Drop the Utilities column

df_alldata.drop('Utilities', inplace=True, axis=1)
df_alldata['Functional'] = df_alldata.fillna('Typ')
df_alldata['LotFrontage'] = df_alldata.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
#Review for any missing data again

alldata_na = df_alldata.isnull().sum()

alldata_na = alldata_na[alldata_na>0]

print('Cols with Missing Values: ',len(alldata_na))
# Transform some numeric col to categorical as they were

numtocatg = ['MSSubClass','OverallCond','YrSold','MoSold']



for col in numtocatg:

    df_alldata[col] = df_alldata[col].astype(str)
#Adding one more feature

df_alldata['TotalSF'] = df_alldata['TotalBsmtSF'] + df_alldata['1stFlrSF'] + df_alldata['2ndFlrSF']
#Dataset with all cols

df_dataset = df_alldata.copy()
#Get Categorical col names

catg_cols = df_dataset.dtypes[df_dataset.dtypes=='object'].index

catg_cols
df_dataset.shape
#Perform One Hot Encoding without dummy variable trap

for col in catg_cols:

    df_temp = df_dataset[col] #Get Col to OHE

    df_temp = pd.get_dummies(df_temp, prefix=col)

    tmp = df_temp.columns[0]

    df_temp.drop(tmp, inplace=True, axis=1) #Drop a dummy variable

    df_dataset = pd.concat([df_dataset, df_temp], axis=1) #Concate OHE cols to original DF

    df_dataset.drop(col, inplace=True, axis=1) #Drop OHE column
df_dataset.shape
df_dataset.head()
df_train = df_dataset[:ntrain]

df_test = df_dataset[ntrain:]
df_train.shape, df_test.shape
y_train.shape
import xgboost as xgb

from xgboost import XGBRegressor
model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=500, n_jobs=-1, gamma=0.1)
model_xgb.fit(df_train, y_train)
prediction = model_xgb.predict(df_test)
prediction[:5]
srs_TestIds = test_copy['Id']
df_submission = pd.DataFrame({'Id':test_copy['Id'], 'SalePrice':prediction})

df_submission.head()
df_submission.to_csv('Submission_xgb_v3.csv', index=False)