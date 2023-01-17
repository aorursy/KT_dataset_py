# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.ensemble import GradientBoostingRegressor

import copy

import warnings

warnings.filterwarnings('ignore')
train_d=pd.read_csv("../input/train.csv")

test_d=pd.read_csv("../input/test.csv")

subm_d=pd.read_csv("../input/sample_submission.csv")
print("Train data shape :",train_d.shape)

print("Test data shape :",test_d.shape)

print("Subm data shape :",subm_d.shape)
train_d.head()
SalePrice =train_d["SalePrice"]

Id=test_d["Id"]
#train_d.drop(["SalePrice"],axis=1,inplace=True)

#print(train_d.shape)
#train_objs_num = len(train_d)

#data = pd.concat(objs=[train_d, test_d], axis=0)

#dataset = pd.get_dummies(dataset)

#train = copy.copy(dataset[:train_objs_num])

#test = copy.copy(dataset[train_objs_num:])
train_d.columns
train_d.describe()
train_d.info()
mis_per=(train_d.isnull().sum()*100/len(train_d)).sort_values(ascending=False)

mis_per.head(15)
mis_per[mis_per>20]
train_d.drop(["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"],axis=1,inplace=True)
plt.figure(figsize=(30,22))

sns.heatmap(train_d.corr())
plt.scatter(train_d["TotalBsmtSF"],train_d["SalePrice"])
plt.scatter(train_d["1stFlrSF"],train_d["SalePrice"])

train_d.drop(["1stFlrSF"],axis=1,inplace=True)

test_d.drop(["1stFlrSF"],axis=1,inplace=True)
plt.scatter(train_d["GarageCars"],train_d["SalePrice"])

train_d.drop(["GarageCars"],axis=1,inplace=True)

test_d.drop(["GarageCars"],axis=1,inplace=True)
plt.scatter(train_d["GarageArea"],train_d["SalePrice"])
plt.scatter(train_d["YearBuilt"],train_d["SalePrice"])
train_d.drop(train_d[train_d["SalePrice"]>700000].index,inplace=True)
plt.scatter(train_d["LotArea"],train_d["SalePrice"])
train_d.drop(train_d[train_d["LotArea"]>150000].index,inplace=True)
sns.countplot(train_d["Utilities"])

#Better dropping it 
train_d = train_d.drop(["Utilities"], axis=1)

test_d = test_d.drop(["PoolQC","Fence", "MiscFeature", "Utilities","Alley","FireplaceQu"], axis=1)
sns.distplot(train_d["SalePrice"])
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea','TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_d[cols], size = 2.5)

plt.show();
var = 'GrLivArea'

data = pd.concat([train_d['SalePrice'],train_d[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train_d.drop(train_d[train_d["GrLivArea"]>4000].index,inplace=True)
var = 'GrLivArea'

data = pd.concat([train_d['SalePrice'],train_d[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(train_d["GrLivArea"])
train_d["GrLivArea"]=np.log(train_d["GrLivArea"])

sns.distplot(train_d["GrLivArea"])
sns.distplot(train_d["LotArea"])

train_d["LotArea"]=np.log(train_d["LotArea"])
sns.distplot(train_d["LotArea"])
print(train_d["SalePrice"].skew())


train_d["SalePrice"]=np.log(train_d["SalePrice"])
sns.distplot(train_d["SalePrice"],fit=norm)
col=train_d.loc[:, train_d.dtypes == np.float64].columns.tolist()

print(col)
for i in col:

    print(train_d[i].skew())
train_d["MasVnrArea"].isnull().sum()
train_d["MasVnrArea"]=train_d["MasVnrArea"].fillna(0)

train_d["MasVnrType"]=train_d["MasVnrType"].fillna("None")



test_d["MasVnrArea"]=test_d["MasVnrArea"].fillna(0)

test_d["MasVnrType"]=test_d["MasVnrType"].fillna("None")
train_d = train_d.fillna({"BsmtQual": "None"})

test_d = test_d.fillna({"BsmtQual": "None"})

train_d = train_d.fillna({"BsmtCond": "None"})

test_d = test_d.fillna({"BsmtCond": "None"})

train_d = train_d.fillna({"BsmtExposure": "None"})

test_d = test_d.fillna({"BsmtExposure": "None"})

train_d = train_d.fillna({"BsmtFinType1": "None"})

test_d = test_d.fillna({"BsmtFinType1": "None"})

train_d = train_d.fillna({"BsmtFinSF1": 0})

test_d = test_d.fillna({"BsmtFinSF1": 0})

train_d = train_d.fillna({"BsmtFinType2": "None"})

test_d = test_d.fillna({"BsmtFinType2": "None"})

test_d = test_d.fillna({"BsmtFinSF2": 0})

test_d = test_d.fillna({"BsmtUnfSF": 0})

test_d = test_d.fillna({"TotalBsmtSF": 0})

test_d = test_d.fillna({"BsmtFullBath": 0})

test_d = test_d.fillna({"BsmtHalfBath": 0})
# Garage

train_d = train_d.fillna({"GarageType": "None"})

test_d = test_d.fillna({"GarageType": "None"})

train_d = train_d.fillna({"GarageYrBlt": 0})

test_d = test_d.fillna({"GarageYrBlt": 0})

train_d = train_d.fillna({"GarageFinish": "None"})

test_d = test_d.fillna({"GarageFinish": "None"})

test_d = test_d.fillna({"GarageArea": 0})

train_d = train_d.fillna({"GarageQual": "None"})

test_d = test_d.fillna({"GarageQual": "None"})

train_d = train_d.fillna({"GarageCond": "None"})

test_d = test_d.fillna({"GarageCond": "None"})
test_d = test_d.fillna({"MSZoning": "RL"})

test_d = test_d.fillna({"Exterior1st": "VinylSd"})

test_d = test_d.fillna({"Exterior2nd": "VinylSd"})

train_d = train_d.fillna({"Electrical": "SBrkr"})

test_d = test_d.fillna({"KitchenQual": "TA"})

test_d = test_d.fillna({"Functional": "Typ"})

test_d = test_d.fillna({"SaleType": "WD"})
train_d = train_d.fillna({"LotFrontage": 0})

test_d = test_d.fillna({"LotFrontage": 0})
Price=train_d["SalePrice"]
train_d.drop(["SalePrice"],axis=1,inplace=True)

#train_d["SalePrice"]
train_d.drop(["Id"],axis=1,inplace=True)

test_d.drop(["Id"],axis=1,inplace=True)
train_d.head()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_objs_num = len(train_d)

data = pd.concat(objs=[train_d, test_d], axis=0)

data = pd.get_dummies(data)

train = copy.copy(data[:train_objs_num])

test = copy.copy(data[train_objs_num:])
train.head()
train=sc.fit_transform(train)
test=sc.fit_transform(test)
from sklearn.decomposition import PCA
pc=PCA(n_components=2)

train=pc.fit_transform(train)

test=pc.fit_transform(test)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
GBoost.fit(train,Price)
GBoost.score(train,Price)
y_pred=GBoost.predict(test)

y_pred=np.exp(y_pred)
y_pred
my_submission = pd.DataFrame({"Id":Id,"SalePrice": y_pred})

print(my_submission)



my_submission.to_csv('submission12.csv', encoding='utf-8', index=False)