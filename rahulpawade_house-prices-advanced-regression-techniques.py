# rmsle-->0.13-0.14



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.shape,test.shape

df = pd.DataFrame()

df = pd.concat([train,test],axis=0)
df.shape
sns.heatmap(df.isna(),yticklabels=0)
df.info()
df = df.drop(columns=['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature','GarageYrBlt',],axis=1)

df.shape
df.info()



df_object = df.select_dtypes(include=object)

df_num = df.select_dtypes(include=np.number)
df_object.info()
for i in df_object.columns:

    if df_object[i].isnull().sum()!=0:

        df_object[i]=df_object[i].fillna(df_object[i].mode()[0])

df_object.head()

df_object['MSZoning'].value_counts()
df_object.columns
df_object = pd.get_dummies(df_object,drop_first=True)

df_object.head()
df_object.shape
for i in df_num.columns:

    if df_num[i].isnull().sum()!=0:

        df_num[i]=df_num[i].fillna(df_num[i].mean())

d = pd.DataFrame()
d = pd.concat([df_object,df_num],axis=1)
d.head()
d.shape
sns.heatmap(d.isnull(),yticklabels=0)
d.head()
x_train = d.iloc[0:1460,:-1]

x_test = d.iloc[1460:,:-1]
y_train = d.iloc[0:1460,-1:]
x_train.shape,x_test.shape,y_train.shape
from xgboost import XGBRegressor

model = XGBRegressor()



from sklearn.feature_selection import SelectFromModel

m = SelectFromModel(model.fit(x_train,y_train), prefit=True)

x_train = m.transform(x_train)

x_test = m.transform(x_test)



model.fit(x_train,y_train)

y_pred = model.predict(x_test)
f = {'Id':test['Id'],'SalePrice':y_pred}

f = pd.DataFrame(f)

f.to_csv('submited_x.csv',index=False)
f.head()