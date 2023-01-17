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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.shape
test_df.shape
df['MSZoning'].value_counts()
df.isnull().sum()
df.isnull().values.any()
df.columns[df.isnull().any()]
test_df.columns[test_df.isnull().any()]
test_df.drop(['Alley'],axis=1,inplace=True)

test_df.drop(['GarageYrBlt'],axis=1,inplace=True)

test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

test_df.drop(['Id'],axis=1,inplace=True)
test_df.shape


test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())

test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])



test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])

test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])

test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])

test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])

test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])



test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])

test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])

test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])

test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
test_df.loc[:, test_df.isnull().any()].head()
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])

test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])

test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])

test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])

test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())

test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())

test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())

test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())

test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])

test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])

test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])

test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])

test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())

test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())

test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df.shape

test_df.to_csv('formulatedtest.csv',index=False)
df.drop(['Alley'],axis=1,inplace=True)

df.drop(['GarageYrBlt'],axis=1,inplace=True)

df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df.drop(['Id'],axis=1,inplace=True)

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])





df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])



df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])



df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])



df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])



df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])



df.dropna(inplace=True)
df.shape
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
def category_onehot_multcols(columns):

    df_final=final_df

    i=0

    for fields in columns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
main_df=df.copy()


## Combine Test Data 



test_df=pd.read_csv('formulatedtest.csv')


## Combine Test Data 



test_df.shape
final_df=pd.concat([df,test_df],axis=0)

final_df['SalePrice']


final_df=pd.concat([df,test_df],axis=0)
final_df['SalePrice']
final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
df_Train=final_df.iloc[:1422,:]

df_Test=final_df.iloc[1422:,:]
df_Train.shape
df_Test.shape


df_Test.drop(['SalePrice'],axis=1,inplace=True)
df_Test


X_train=df_Train.drop(['SalePrice'],axis=1)

y_train=df_Train['SalePrice']
X_train
y_train
import xgboost

classifier=xgboost.XGBRegressor()

classifier.fit(X_train,y_train)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(classifier, open(filename, 'wb'))
y_pred=classifier.predict(df_Test)
y_pred
##Create Sample Submission file and Submit using Xboost

pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)