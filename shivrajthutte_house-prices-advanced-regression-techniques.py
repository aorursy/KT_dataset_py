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
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
data=pd.concat([train,test],sort=True)

data=data.drop(["Alley","YearBuilt","YearRemodAdd","GarageYrBlt",

            "PoolQC","Fence","MiscFeature","YrSold"],axis=1)

mean_value=data['LotFrontage'].mean()

data['LotFrontage']=data['LotFrontage'].fillna(mean_value)



mean_value=data['GarageArea'].mean()

data['GarageArea']=data['GarageArea'].fillna(mean_value)



mean_value=data['TotalBsmtSF'].mean()

data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(mean_value)



median_value=data['MasVnrArea'].median()

data['MasVnrArea']=data['MasVnrArea'].fillna(median_value)



median_value=data['BsmtFinSF1'].median()

data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(median_value)



median_value=data['BsmtFinSF2'].median()

data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(median_value)



median_value=data['BsmtUnfSF'].median()

data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(median_value)



median_value=data['SalePrice'].median()

data['SalePrice']=data['SalePrice'].fillna(median_value)
education1=['Electrical','MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType',

            'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

           'BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','FireplaceQu',

           'GarageType','GarageCars','GarageFinish','GarageQual','GarageCond','SaleType']

           

data[education1]=data[education1].fillna(data.mode().iloc[0])
data=pd.get_dummies(data,drop_first=True)

train1=data[0:1460]

test1=data[1460:2919]

train1=train1.drop(["Id"],axis=1)

X=train1.iloc[:,train1.columns!='SalePrice']

y=train1.iloc[:,train1.columns=='SalePrice']

test1=test1.drop(["SalePrice"],axis=1)

TestID = test1["Id"]

Id = pd.DataFrame(TestID)
test_final=test1.iloc[:,test1.columns!='Id']
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=500,

                       n_jobs=None, oob_score=True, random_state=1112, verbose=0,

                       warm_start=False)
model_rf.fit( X, y )

y_pred_rf=model_rf.predict(test_final)
y_pred_rf
Id = Id['Id']

SalePrice = y_pred_rf



submit = pd.DataFrame({'Id':Id, 'SalePrice':SalePrice})