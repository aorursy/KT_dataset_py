# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import IPython

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy.stats import norm

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test=pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/train.csv')

test.head()

#test.shape

#(1459,80)
print(train.shape)

train.head()
train_id=train['Id']

test_id=test['Id']

train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)
train.SalePrice.describe()
sns.distplot(train['SalePrice'],fit=norm)

plt.xlabel('SalePrice')

plt.ylabel('Freq')

plt.title('SalePrice Distribution')
train['SalePrice']=np.log1p(train.SalePrice)

sns.distplot(train['SalePrice'],fit=norm)

plt.xlabel('log(1+SalePrice)')

plt.ylabel('Freq')

plt.title('log(1+SalePrice) Distribution')
ntrain=train.shape[0]

ntest=test.shape[0]

y_train=train['SalePrice']

data= pd.concat((train,test),sort=True).reset_index(drop=True)

heat=data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(heat,vmax=0.9,square=True)
train_rows=train.shape[0]

test_rows=test.shape[0]

data.drop(['SalePrice'],axis=1,inplace=True)

data.isna().sum().sort_values(ascending=False)
cols_drop=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageFinish','GarageQual','GarageYrBlt','GarageCond','GarageType']

for col in cols_drop:

    data.drop(col,axis=1,inplace=True)

data.shape
data['MSSubClass'] = data['MSSubClass'].apply(str)





data['OverallCond'] = data['OverallCond'].astype(str)







data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
data.drop('Utilities',axis=1,inplace=True)
data.columns
from sklearn.preprocessing import LabelEncoder
#No Basement

data.BsmtQual =data.BsmtQual.fillna("NONE")

data.BsmtCond = data.BsmtCond.fillna("NONE")

data.BsmtFinType1=data.BsmtFinType1.fillna("NONE")

data.BsmtFinType2=data.BsmtFinType2.fillna("NONE")

data.BsmtFinSF1=data.BsmtFinSF1.fillna(0)

data.BsmtFinSF2=data.BsmtFinSF2.fillna(0)

data.BsmtExposure=data.BsmtExposure.fillna("NONE")

data.TotalBsmtSF=data.TotalBsmtSF.fillna(0)

data.BsmtFullBath=data.BsmtFullBath.fillna(0)

data.BsmtHalfBath=data.BsmtHalfBath.fillna(0)

data.BsmtUnfSF=data.BsmtUnfSF.fillna(0)

#NoGarage

data.GarageCars=data.GarageCars.fillna(0)

data.GarageArea=data.GarageArea.fillna(0)

#No Kitchen

data.KitchenQual=data.KitchenQual.fillna("NONE")

#functionality

data.Functional=data.Functional.fillna("Typ")

#Electrical. Most values are SBrkr

data.Electrical=data.Electrical.fillna("SBrkr")

#Exterior1st and Exterior2nd

data.Exterior1st=data.Exterior1st.fillna(data.Exterior1st.mode()[0])

#No Masonary Veneer Area

data.MasVnrArea=data.MasVnrArea.fillna(0)

data.MasVnrType=data.MasVnrType.fillna("NONE")

#MSZoning

data.MSZoning=data.MSZoning.fillna(data.MSZoning.mode()[0])

#SaleType

data.SaleType=data.SaleType.fillna(data.SaleType.mode()[0])
cols = ( 'BsmtQual', 'BsmtCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure','LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold','1stFlrSF', '2ndFlrSF','TotalBsmtSF')

for col in cols:

    label=LabelEncoder()

    data[col]=label.fit_transform(list(data[col].values))

print(data.shape)

data=pd.get_dummies(data)

data.shape
data.head()
num_features=data.dtypes[data.dtypes!="object"].index
#separating train and test dataset

train= data[:ntrain]

test=data[ntrain:]
dict(data.isna().sum())
import xgboost as xgb

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error
folds=4



def cross_v(model):

    k_fold=KFold(folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv=k_fold))

    return rmse
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
elastic_net= make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
lasso_score= cross_v(lasso)

print("Lasso Score={:.4f} ".format(lasso_score.mean()))
enet_score=cross_v(elastic_net)

print("Elastic Net Score={:.4f} ".format(enet_score.mean()))
xgb_score=cross_v(model_xgb)

print("XGBoost Score={:.4f} ".format(xgb_score.mean()))
def rmse(y,pred):

    return np.sqrt(mean_squared_error(y,pred))
model_xgb.fit(train,y_train)

train_pred= model_xgb.predict(train)

final_pred=np.expm1(model_xgb.predict(test))

print("Final rmse Score= {:.4f}".format(rmse(y_train,train_pred)))
submission=pd.DataFrame()

submission['Id']=test_id

submission['SalePrice']=final_pred

submission.to_csv('submission.csv',index=False)