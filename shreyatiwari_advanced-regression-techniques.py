# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()
Df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
Df.head()
Df.shape

Df.ndim
Df.columns
Df.describe
Df.isnull().sum()
df.shape
df.columns
df.dtypes
df.ndim
df.describe
df.isnull().sum()
df.isna().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.info()
df['LotFrontage'].fillna(df['LotFrontage'].mean())
Df['LotFrontage'].fillna(df['LotFrontage'].mean())
Df['MSZoning'].value_counts()
Df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['MSZoning'].value_counts()
df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
Df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
Df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'].fillna(df['GarageType'].mode()[0])
Df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish'].fillna(df['GarageType'].mode()[0])
Df['GarageFinish'].fillna(df['GarageType'].mode()[0])
df['GarageQual'].fillna(df['GarageType'].mode()[0])
Df['GarageQual'].fillna(df['GarageType'].mode()[0])
df['GarageCond'].fillna(df['GarageType'].mode()[0])
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
Df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea'].fillna(df['MasVnrType'].mode()[0])
Df['MasVnrArea'].fillna(df['MasVnrType'].mode()[0])
df.drop(['Alley'],axis=1,inplace=True)
Df.drop(['Alley'],axis=1,inplace=True)
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
Df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df.drop(['GarageYrBlt'],axis=1,inplace=True)
Df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['Id'],axis=1,inplace=True)
Df.drop(['Id'],axis=1,inplace=True)
df.shape
Df.shape
Df.isnull().sum()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
sns.heatmap(Df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='BuGn')
df.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)
Df.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)
sns.heatmap(Df.isnull(),yticklabels=False,cbar=False,cmap='BuGn')
df.dropna(inplace=True)
Df.dropna(inplace=True)
df.shape
Df.shape
df.head()
Df.head()
df.columns
Df.columns
def Category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df(fields),drop_first=True)
        
        final_df.drop([feilds],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
        
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final
        
        
Df.to_csv('formulatedtest.csv',index=False)
main_df=df.copy()
test_df=pd.read_csv("formulatedtest.csv")
test_df.shape
test_df.head()
final_df=pd.concat([df,test_df],axis=0)
final_df.shape
final_df.head()
final_df.loc[:,~final_df.columns.duplicated()]
df_train=final_df.iloc[:572,:]
df_test=final_df.iloc[:572,:]
df_test.drop(['SalePrice'],axis=1,inplace=True)
df_test.shape
x_train=df_train.drop(['SalePrice'],axis=1)
y_train=df_train['SalePrice']
import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(x_train,y_train)
import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier,open(filename,'wb'))
y_pred=classifier.predict(df_test)
y_pred
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.to_csv('sample_submission.csv',index=False)