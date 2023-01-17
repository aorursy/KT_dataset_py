import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/mercedes-benz-greener-manufacturing/train.csv.zip")
train.info()
train.shape
test = pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")
test.shape
df = pd.concat([train,test],axis=0)
df.shape
df.describe()
sns.heatmap(df.isna(),yticklabels=0)
df_obj = df.select_dtypes(include=object)

df_num = df.select_dtypes(include=np.number)
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
for i in df_obj.columns:

    df_obj[i]=l.fit_transform(df_obj[i])
d = pd.DataFrame()

d = pd.concat([df_obj,df_num],axis=1)
df_train = d.iloc[:4209,:]
d.drop(columns=['ID','y'],axis=1,inplace=True)
x_train = d.iloc[:4209,:]

x_test = d.iloc[4209:,:]

y_train = train['y']
from xgboost import XGBRegressor

m = XGBRegressor()
from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(m.fit(x_train,y_train), prefit=True)

x_train = model.transform(x_train)

x_test = model.transform(x_test)
x_train.shape,x_test.shape,y_train.shape
m.fit(x_train,y_train)
y_pred = m.predict(x_test)
f = {"ID":test["ID"],"y":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)
f.head()