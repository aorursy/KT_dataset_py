# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.pandas.set_option('display.max_columns',None)
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.shape
df.describe()
df.isnull().sum().head(60)
df.isnull().sum().tail(21)
plt.figure(figsize=(14,14))

sns.heatmap(df.isna(),cbar=False)
# print(df.LotFrontage.isna().sum())

df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage.mean()) 

df.LotFrontage.isna().sum()
df = df.drop(['Id', 'Alley','PoolQC','Fence', 'MiscFeature'],axis= 1)
df.FireplaceQu = df.FireplaceQu.fillna(df.FireplaceQu.mode()[0])

# df.FireplaceQu.isna().sum()
df[['BsmtQual','BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2']].info()

df.BsmtQual = df.BsmtQual.fillna(df.BsmtQual.mode()[0])

df.BsmtCond = df.BsmtCond.fillna(df.BsmtCond.mode()[0])

df.BsmtExposure = df.BsmtExposure.fillna(df.BsmtExposure.mode()[0])

df.BsmtFinType1 = df.BsmtFinType1.fillna(df.BsmtFinType1.mode()[0])

df.BsmtFinSF1 = df.BsmtFinSF1.fillna(df.BsmtFinSF1.mean())

df.BsmtFinType2 = df.BsmtFinType2.fillna(df.BsmtFinType2.mode()[0])



df[['GarageType','GarageYrBlt', 'GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].info()

df.GarageType = df.GarageType.fillna(df.GarageType.mode()[0])

df.GarageYrBlt = df.GarageYrBlt.fillna(round(df.GarageYrBlt.mean(),1))

df.GarageFinish = df.GarageFinish.fillna(df.GarageFinish.mode()[0])

df.GarageCars = df.GarageCars.fillna(round(df.GarageCars.mean()))

df.GarageArea = df.GarageArea.fillna(round(df.GarageArea.mean()))

df.GarageQual = df.GarageQual.fillna(df.GarageQual.mode()[0])

df.GarageCond = df.GarageCond.fillna(df.GarageCond.mode()[0])





# df[['MasVnrType', 'MasVnrArea']].info()

df.MasVnrType = df.MasVnrType.fillna(df.MasVnrType.mode()[0])

df.MasVnrArea = df.MasVnrArea.fillna(round(df.MasVnrArea.mean(),1))

df[['MasVnrType', 'MasVnrArea']].isna().sum()
df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0])

df.isna().sum()
plt.figure(figsize=(14,14))

sns.heatmap(df.isna(),cbar=False)
df.isna().sum().sum()
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
test_df.isnull().sum().head(60)
test_df.isnull().sum().tail(21)
test_df.BsmtQual = test_df.BsmtQual.fillna(test_df.BsmtQual.mode()[0])

test_df.BsmtCond = test_df.BsmtCond.fillna(test_df.BsmtCond.mode()[0])

test_df.BsmtExposure = test_df.BsmtExposure.fillna(test_df.BsmtExposure.mode()[0])

test_df.BsmtFinType1 = test_df.BsmtFinType1.fillna(test_df.BsmtFinType1.mode()[0])

test_df.BsmtFinSF1 = test_df.BsmtFinSF1.fillna(test_df.BsmtFinSF1.mean())

test_df.BsmtFinType2 = test_df.BsmtFinType2.fillna(test_df.BsmtFinType2.mode()[0])

test_df.GarageType = test_df.GarageType.fillna(test_df.GarageType.mode()[0])

test_df.GarageYrBlt = test_df.GarageYrBlt.fillna(round(test_df.GarageYrBlt.mean(),1))

test_df.GarageFinish = test_df.GarageFinish.fillna(test_df.GarageFinish.mode()[0])

test_df.GarageCars = test_df.GarageCars.fillna(round(test_df.GarageCars.mean()))

test_df.GarageArea = test_df.GarageArea.fillna(round(test_df.GarageArea.mean()))

test_df.GarageQual = test_df.GarageQual.fillna(test_df.GarageQual.mode()[0])

test_df.GarageCond = test_df.GarageCond.fillna(test_df.GarageCond.mode()[0])

test_df.MasVnrType = test_df.MasVnrType.fillna(test_df.MasVnrType.mode()[0])

test_df.MasVnrArea = test_df.MasVnrArea.fillna(round(test_df.MasVnrArea.mean(),1))

test_df.Electrical = test_df.Electrical.fillna(test_df.Electrical.mode()[0])

test_df = test_df.drop(['Id', 'Alley','PoolQC','Fence', 'MiscFeature'],axis= 1)

test_df.LotFrontage = test_df.LotFrontage.fillna(test_df.LotFrontage.mean())

test_df.FireplaceQu = test_df.FireplaceQu.fillna(test_df.FireplaceQu.mode()[0])

test_df.MSZoning = test_df.MSZoning.fillna(test_df.MSZoning.mode()[0])

test_df.Utilities = test_df.Utilities.fillna(test_df.Utilities.mode()[0])

test_df.Exterior1st = test_df.Exterior1st.fillna(test_df.Exterior1st.mode()[0])

test_df.Exterior2nd = test_df.Exterior2nd.fillna(test_df.Exterior2nd.mode()[0])

test_df.BsmtFinSF2 = test_df.BsmtFinSF2.fillna(test_df.BsmtFinSF2.mode()[0])

test_df.BsmtUnfSF = test_df.BsmtUnfSF.fillna(test_df.BsmtUnfSF.mode()[0])

test_df.BsmtFullBath = test_df.BsmtFullBath.fillna(test_df.BsmtFullBath.mode()[0])

test_df.BsmtHalfBath = test_df.BsmtHalfBath.fillna(test_df.BsmtHalfBath.mode()[0])

test_df.KitchenQual = test_df.KitchenQual.fillna(test_df.KitchenQual.mode()[0])

test_df.TotRmsAbvGrd = test_df.TotRmsAbvGrd.fillna(test_df.TotRmsAbvGrd.mode()[0])

test_df.Functional = test_df.Functional.fillna(test_df.Functional.mode()[0])

test_df.SaleType = test_df.SaleType.fillna(test_df.SaleType.mode()[0])

test_df.TotalBsmtSF = test_df.TotalBsmtSF.fillna(round(test_df.TotalBsmtSF.mean()))

test_df.BsmtFinSF2.dtype

test_df.BsmtFinSF2.value_counts()
test_df.isnull().sum().head(60)
test_df.isnull().sum().tail(21)
test_df.isnull().sum().sum()
plt.figure(figsize=(14,14))

sns.heatmap(test_df.isna(),cbar=False)
col = df.columns.to_series().groupby(df.dtypes).groups

col
# handling catagorical data



col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

        'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

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
df.shape
test_df.shape
final_df=pd.concat([df,test_df],axis=0)
final_df.shape
final_df.SalePrice
final_df=category_onehot_multcols(col)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df.head()
df_train= final_df.iloc[:1460,:]

df_test = final_df.iloc[1460:,:]
df_test= df_test.drop(['SalePrice'],axis=1)

df_test.shape
df_train.shape
df_test.shape
X= df_train.drop(['SalePrice'],axis=1)

y= df_train['SalePrice']

X.shape
y.shape
from xgboost import XGBRFRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error





# print(X_train.shape, X_test.shape)
classifier= XGBRFRegressor()

classifier.fit(X,y)
y_pred = classifier.predict(df_test)



y_pred
pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)