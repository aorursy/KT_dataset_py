# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# made by songwonmin

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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
train.columns
sample_submission
train.describe(include="all")
train.info()
train.drop(['Alley','PoolQC','Fence','MiscFeature',"FireplaceQu"],axis=1)
feature_names = ["Id","MSSubClass","MSZoning","LotArea","Street","Condition1","BldgType","OverallQual","OverallCond","YearBuilt","YearRemodAdd","ExterCond","BsmtCond","TotalBsmtSF","1stFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd","GarageYrBlt","GarageArea","MiscVal","YrSold","SaleType","SaleCondition"]

X = train[feature_names]

test = test[feature_names]
object_columns = X.select_dtypes(include=['object'])

numerical_columns =X.select_dtypes(exclude=['object'])
object_columns.dtypes
numerical_columns.dtypes
sns.barplot(x="MSZoning",y="SalePrice",data=train)

# MSZoning: 일반 구역 구분

# 물위에 있는 집이 가장 비싸고 주거용으로 저밀도인 지역이 가격이 높다.
X.MSZoning.sample(10)
MSZoning_group_mapping={"RL":0,"RM":1,"C (all)":2,"FV":3,"RH":4}

X["MSZoning"]=X["MSZoning"].map(MSZoning_group_mapping)

test["MSZoning"]=test["MSZoning"].map(MSZoning_group_mapping)
sns.barplot(x="Street",y="SalePrice",data=train)

# Street: 도로 접근 유형 -> 비포장 도로 보단, 포장도로일수록 가격이 높다는 것을 알 수 있다
Street_group_mapping={"Pave":0,"Grvl":1}

X["Street"]=X["Street"].map(Street_group_mapping)

test["Street"]=test["Street"].map(Street_group_mapping)
sns.barplot(x="Condition1",y="SalePrice",data=train)

# Condition1: 간선도로 또는 철도와의 근접성

# 공원이 근처에 있고 철도가 바로 옆에 있는 것 보다 200호이내 있을 때 가격이 높다.
Condition1_group_mapping={"Norm":0,"Feedr":1,"PosN":2,"Artery":3,"RRAe":4,"RRNn":5,"RRAn":6,"PosA":7,"RRNe":8}

X["Condition1"]=X["Condition1"].map(Condition1_group_mapping)

test["Condition1"]=test["Condition1"].map(Condition1_group_mapping)
X.isnull().sum()
sns.barplot(x="BldgType",y="SalePrice",data=train)

# BldgType: 주거 유형 -> 1인 가구와 Townhouse End Unit의 가격이 비싸고 나머지는 비슷하다.
X.BldgType.sample(10)
BldgType_group_mapping={"1Fam":0,"2fmCon":1,"Duplex":2,"TwnhsE":3,"Twnhs":4}

X["BldgType"]=X["BldgType"].map(BldgType_group_mapping)

test["BldgType"]=test["BldgType"].map(BldgType_group_mapping)
sns.barplot(x="ExterCond",y="SalePrice",data=train)

# ExterCond: 외관 -> 평균 이상이면 가격 변동은 크지 않지만, 평균 미만일 경우 가격 하락폭이 크다.
ExterCond_group_mapping={"TA":0,"Gd":1,"Fa":2,"Po":3,"Ex":4}

X["ExterCond"]=X["ExterCond"].map(ExterCond_group_mapping)

test["ExterCond"]=test["ExterCond"].map(ExterCond_group_mapping)
sns.barplot(x="BsmtCond",y="SalePrice",data=train)

# BsmtCond : 지하실의 일반상태

# 지하실의 상태가 좋을수록 가격이 높다.
BsmtCond_group_mapping={"TA":0,"Gd":1,"Fa":2,"Po":3}

X["BsmtCond"]=X["BsmtCond"].map(BsmtCond_group_mapping)

test["BsmtCond"]=test["BsmtCond"].map(BsmtCond_group_mapping)
sns.barplot(x="SaleType",y="SalePrice",data=train)

#SaleType: 판매 유형

#새로 지은 집일 때 가격이 비쌌고 저금리나 현금으로 구매 할 때도 가격이 비쌌다.
SaleType_group_mapping={"WD":0,"New":1,"COD":2,"ConLD":3,"ConLi":4,"CWD":5,"ConLw":6,"Con":7,"Oth":8}

X["SaleType"]=X["SaleType"].map(SaleType_group_mapping)

test["SaleType"]=test["SaleType"].map(SaleType_group_mapping)
sns.barplot(x="SaleCondition",y="SalePrice",data=train)

# SaleCondition: 판매조건

# 새로 지은 집이 가장 비싸다.
SaleCondition_group_mapping={"Normal":0,"Abnorml":1,"Partial":2,"AdjLand":3,"Alloca":4,"Family":5}

X["SaleCondition"]=X["SaleCondition"].map(SaleCondition_group_mapping)

test["SaleCondition"]=test["SaleCondition"].map(SaleCondition_group_mapping)
sns.barplot(x="MSSubClass",y="SalePrice",data=train)
sns.barplot(x="LotArea",y="SalePrice",data=train)
sns.barplot(x="OverallQual",y="SalePrice",data=train)

#OverallQual: 전체재료 및 마감 품질

#마감 품질이 높을수록 가격이 높다는 것을 알 수 있다.
sns.barplot(x="OverallCond",y="SalePrice",data=train)

# OverallCond: 전반적인 상태 등급 -> 2, 5, 9등급의 집값이 주변 등급 대비 높으며, 6~8등급의 집값은 비슷하다.
sns.barplot(x="YearBuilt",y="SalePrice",data=train)
X["YearBuilt"].describe()
X["YearBuilt"]
YearBuilt_slice=np.linspace(1870,2010,15)
YearBuilt_slice
YearBuilt_labels=np.arange(0,14)
YearBuilt_labels
X["YearBuiltGroup"]=pd.cut(X["YearBuilt"],YearBuilt_slice,labels=YearBuilt_labels)

test["YearBuiltGroup"]=pd.cut(test["YearBuilt"],YearBuilt_slice,labels=YearBuilt_labels)
X["YearBuiltGroup"]
sns.barplot(x="YearRemodAdd",y="SalePrice",data=train)
X["YearRemodAdd"].describe()
YearRemodAdd_slice=np.linspace(1950,2010,7)
YearRemodAdd_labels=np.arange(0,6)
X["YearRemodAddGroup"]=pd.cut(X["YearRemodAdd"],YearRemodAdd_slice,labels=YearRemodAdd_labels)

test["YearRemodAddGroup"]=pd.cut(test["YearRemodAdd"],YearRemodAdd_slice,labels=YearRemodAdd_labels)
X["YearRemodAddGroup"]
sns.barplot(x="TotalBsmtSF",y="SalePrice",data=train)
sns.barplot(x="1stFlrSF",y="SalePrice",data=train)
X["1stFlrSF"].describe()
sns.barplot(x="FullBath",y="SalePrice",data=train)

# FullBath: 등급 이상의 전체 욕실

# 욕실의 개수가 많을수록 가격이 높다.
sns.barplot(x="BedroomAbvGr",y="SalePrice",data=train)

# Bedroom: 지하층 이상 침실 수

# 침실개수는 4개일 때와 8개일 때 가격이 높은 걸로 보아 가족 구성원수에 맞게 침실이 있을 때 가격이 높은 것 같다.
sns.barplot(x="TotRmsAbvGrd",y="SalePrice",data=train)

# TotRmsAbvGrd: 총 객실 등급 이상(화장실 포함 안 함) -> 방의 수가 11개까지는 가격이 증가하고 이후로는 감소한다.
sns.barplot(x="GarageYrBlt",y="SalePrice",data=train)
sns.barplot(x="GarageArea",y="SalePrice",data=train)
sns.barplot(x="MiscVal",y="SalePrice",data=train)

# MiscVal: 기타 기능의 $값  -> 기타 기능의 값과 집값의 연관성은 크게 없다
sns.barplot(x="YrSold",y="SalePrice",data=train)

# YrSold: 판매 연도 -> 판매 연도와 짒갑의 연관성은 없다.
blacklist = ["LotArea","TotalBsmtSF","1stFlrSF","GarageYrBlt","GarageArea","YearBuilt","YearRemodAdd"]
X = X.drop(blacklist,axis=1)

test = test.drop(blacklist,axis=1)
X.info()
y = train["SalePrice"]
X.isnull().sum()
X["SaleType"]=X["SaleType"].notnull().astype("int64")

X["YearBuiltGroup"]=X["YearBuiltGroup"].notnull().astype("int64")

X["YearRemodAddGroup"]=X["YearRemodAddGroup"].notnull().astype("int64")

test["SaleType"]=test["SaleType"].notnull().astype("int64")

test["YearBuiltGroup"]=test["YearBuiltGroup"].notnull().astype("int64")

test["YearRemodAddGroup"]=test["YearRemodAddGroup"].notnull().astype("int64")
X["BsmtCond"]=X["BsmtCond"].fillna("0")

X["YearRemodAddGroup"]=X["YearRemodAddGroup"].fillna("0")



test["MSZoning"]=test["MSZoning"].fillna("0")

test["BsmtCond"]=test["BsmtCond"].fillna("0")

test["YearRemodAddGroup"]=test["YearRemodAddGroup"].fillna("0")
numerical_columns2 = X.select_dtypes(exclude=['object'])

numerical_columnstest = test.select_dtypes(exclude=['object'])
numerical_columns2.dtypes
numerical_columnstest.dtypes
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse(y,preds):

    return sqrt(mean_squared_error(preds, y))
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
ferrari = DecisionTreeRegressor(random_state=1)

ferrari.fit(train_X, train_y)

val_predictions = ferrari.predict(val_X)

val_mae = rmse(val_predictions, val_y)

print("score: {:.0f}".format(val_mae))
porsche = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

porsche.fit(train_X, train_y)

val_predictions = porsche.predict(val_X)

val_mae = rmse(val_y, val_predictions)

print("score: {:.0f}".format(val_mae))
pagani = RandomForestRegressor(random_state=1)

pagani.fit(train_X, train_y)

rf_val_predictions = pagani.predict(val_X)

rf_val_mae = rmse(rf_val_predictions, val_y)



print("score: {:.0f}".format(rf_val_mae))
lamborghini = RandomForestRegressor(random_state=1)

lamborghini.fit(X, y)
sample_submission
test
X
test_preds = lamborghini.predict(test)

output = pd.DataFrame({'Id': test["Id"],

                       'SalePrice': test_preds})

output.to_csv('lamborghini1.csv', index=False)