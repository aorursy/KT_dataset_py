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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

import matplotlib.pyplot as plt

import seaborn as sns

train.head()
train.shape
test.shape
a = train.columns[train.dtypes == object]

b = train.columns[train.dtypes != object]
train.SalePrice.describe()
from scipy.stats import skew,norm

plt.figure(figsize = (8,8))

sns.distplot(train.SalePrice,fit =norm)
train["SalePrice"] = np.log1p(train["SalePrice"])

sns.distplot(train.SalePrice,fit =norm)
from scipy import stats

stats.probplot(train["SalePrice"],plot = plt)
def missing_values(data):

    a = []

    b = []

    for item in data.columns:

        if data[item].isnull().sum()>0:

            a.append(item)

            b.append(data[item].isnull().sum()/(data.shape[0]))

    x = pd.DataFrame()

    x["column_name"] = a

    x["perc_ratio"] = b

    return x
missing_values(train).sort_values(by = "perc_ratio",ascending = False)
missing_values(test).sort_values(by = "perc_ratio",ascending = False)
missing_values(train[a])
missing_values(train[b])
train[b].corr()["SalePrice"].abs().sort_values(ascending = False)
sns.violinplot("OverallQual","SalePrice",data=train)
plt.scatter("GrLivArea","SalePrice",data = train)
train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 12.5)].index,inplace = True)
#outliers above 4000 and linear
train[["GarageCars","GarageArea"]].corr()
#highly correlated(cars and area)
sns.violinplot("GarageCars","SalePrice",data =train)
plt.scatter("GarageArea","SalePrice",data = train)
#outliers above 12000 but present in test set too
plt.scatter("TotalBsmtSF","SalePrice",data = train)
#an outlier detected above 5000
plt.scatter("1stFlrSF","SalePrice",data = train)
#an outlier above 4000
sns.violinplot("FullBath","SalePrice",data = train)
plt.figure(figsize = (20,20))

sns.boxplot("YearBuilt","SalePrice",data=train)

plt.xticks(rotation = 90)

plt.show()

missing_values(train[a]).sort_values(by = "perc_ratio",ascending = False)
missing_values(train[b])
missing_values(test[b.drop("SalePrice")]).sort_values(by = "perc_ratio",ascending = False)
missing_values(test[a]).sort_values(by = "perc_ratio",ascending = False)
train.head()
all_data = pd.concat([train.drop("SalePrice",axis = 1),test])
all_data[all_data["Alley"] == "Pave"]["Neighborhood"].value_counts(normalize=True)
all_data[all_data["Alley"] == "Grvl"]["Neighborhood"].value_counts(normalize = True)
train["Alley"].fillna("None",inplace = True)
test["Alley"].fillna("None",inplace = True)
train["GarageType"].fillna("None",inplace = True)

test["GarageType"].fillna("None",inplace = True)
train["MiscFeature"].fillna("None",inplace = True)

test["MiscFeature"].fillna("None",inplace = True)
train.head()
train["FireplaceQu"].fillna("none",inplace = True)

test["FireplaceQu"].fillna("none",inplace = True)
train["GarageFinish"].fillna("none",inplace = True)

test["GarageFinish"].fillna("none",inplace = True)
train["GarageCond"].fillna("none",inplace = True)

train["GarageQual"].fillna("none",inplace = True)

test["GarageCond"].fillna("none",inplace = True)

test["GarageQual"].fillna("none",inplace = True)
train["BsmtExposure"].fillna("none",inplace = True)



train["BsmtFinType2"].fillna("none",inplace = True)

test["BsmtExposure"].fillna("none",inplace = True)



test["BsmtFinType2"].fillna("none",inplace = True)
train["BsmtQual"].fillna("none",inplace = True)

train["BsmtCond"].fillna("none",inplace = True)

test["BsmtQual"].fillna("none",inplace = True)

test["BsmtCond"].fillna("none",inplace = True)

train["BsmtFinType1"].fillna("none",inplace = True)

test["BsmtFinType1"].fillna("none",inplace = True)
train["Electrical"].fillna("SBrkr",inplace = True)

test["Electrical"].fillna("SBrkr",inplace = True)
missing_values(train[a])
missing_values(train[b]).sort_values(by = "perc_ratio",ascending = False)
train["MasVnrArea"].fillna(0,inplace = True)

train["MasVnrType"].fillna("None",inplace = True)

test["MasVnrType"].fillna("None",inplace = True)

test["MasVnrArea"].fillna(0,inplace = True)
train["PoolQC"].fillna("None",inplace = True)

test["PoolQC"].fillna("None",inplace = True)
train["GarageYrBlt"].fillna(0,inplace = True)

test["GarageYrBlt"].fillna(0,inplace = True)

test["GarageArea"].fillna(0,inplace = True)

test["GarageCars"].fillna(0,inplace = True)
for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath',

            'BsmtHalfBath'):

    test[col].fillna(0,inplace = True)
test["MSZoning"].value_counts()
test["MSZoning"].fillna("RL",inplace = True)
test["Utilities"].value_counts()
test["Functional"].value_counts()
train["Utilities"].value_counts()
train["Functional"].value_counts()
train.drop("Utilities",axis = 1,inplace = True)

test.drop("Utilities",axis = 1,inplace = True)
train["Functional"].fillna("Typ",inplace = True)

test["Functional"].fillna("Typ",inplace = True)
missing_values(train)

missing_values(test)
test["Exterior2nd"].fillna(-1,inplace = True)
test[test["Exterior2nd"] == -1]
test["Exterior2nd"].replace(-1,"VinylSd",inplace = True)
test["Exterior1st"].fillna("VinylSd",inplace = True)
train["SaleType"].value_counts()
test["SaleType"].fillna("WD",inplace = True)
train["Fence"].fillna("None",inplace = True)

test["Fence"].fillna("None",inplace = True)
test["KitchenQual"].fillna(-1,inplace = True)
test[test["KitchenQual"] == -1]["OverallQual"]
test["KitchenQual"].replace(-1,"TA",inplace = True)
missing_values(test)
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x :x.fillna(x.median()))

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x :x.fillna(x.median()))
columns = train.columns[train.dtypes ==  object]

columns1 = train.columns[train.dtypes !=  object]
columns1 = columns1.drop("SalePrice")
skewness = pd.DataFrame(train[columns1].skew(),columns = ["skewness"])

    

    
skewed_columns = skewness[abs(skewness["skewness"]) > .75].index
skewness[abs(skewness["skewness"]) > .75]
from scipy.special import boxcox1p



for item in skewed_columns:

    train[item] = boxcox1p(train[item],.25)

    test[item] = boxcox1p(test[item],.25)

train.shape
train[columns].head()
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir']
columns.drop(cols)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

for item in cols:

    train[item] = lb.fit_transform(train[item])

    test[item] = lb.transform(test[item])
train = pd.get_dummies(train)

test = pd.get_dummies(test)
a = []

for item in train.columns:

    if item not in test.columns:

        a.append(item)

a.remove("SalePrice")
train.drop(a,axis = 1,inplace = True)
predictor = train.drop("SalePrice",axis = 1)

target = train.SalePrice
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.metrics import mean_squared_error
kf = KFold(5,shuffle = True,random_state = 123).get_n_splits()
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()
def rmsle(model):

    rmse = np.sqrt(-cross_val_score(model,rb.fit_transform(predictor),target,scoring = "neg_mean_squared_error",cv = kf)).mean()

    return rmse
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

lr = LinearRegression()

ls = Lasso(alpha = .0004,random_state=1)

rd = Ridge(alpha = 10,random_state = 1)

el = ElasticNet(alpha =.0008,random_state =1 )
print(rmsle(lr))

print(rmsle(ls))

print(rmsle(rd))

print(rmsle(el))
from sklearn.ensemble import RandomForestRegressor

rnd = RandomForestRegressor(random_state = 123,max_features= .6,max_depth = 15,n_estimators = 50)

print(rmsle(rnd))
from sklearn.kernel_ridge import KernelRidge

kr = KernelRidge(alpha = .3,kernel = "polynomial",degree = 2,coef0=4)

print(rmsle(kr))
from sklearn.ensemble import GradientBoostingRegressor

gr = GradientBoostingRegressor(max_features = .2 ,loss = "huber",random_state=123,max_depth = 4,learning_rate = .05,n_estimators = 3000)
print(rmsle(gr))
lg = lgb.LGBMRegressor(objective = "regression",)
print(rmsle(lg))
ls.fit(rb.fit_transform(predictor),target)
prediction = ls.predict(rb.transform(test))
prediction1 = np.expm1(prediction)
submit = pd.DataFrame()

submit["Id"] = test["Id"]

submit["SalePrice"] = prediction1
submit.to_csv('submission.csv', index = False)