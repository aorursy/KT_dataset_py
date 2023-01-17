import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train = pd.read_csv("../input/allstate-claims-severity/train.csv")
train.head()
train.shape
test = pd.read_csv("../input/allstate-claims-severity/test.csv")
test.head()
df = pd.concat([train,test],axis=0)
df = df.drop(columns=["id","loss"],axis=1)
plt.figure(figsize=(20,20))

sns.heatmap(df.isna(),yticklabels=0)
df_obj = df.select_dtypes(include=object)
df_obj.shape
for i in df_obj.columns:

    df_obj[i]=l.fit_transform(df_obj[i])
df_num = df.select_dtypes(include=np.number)
d = pd.concat([df_obj,df_num],axis=1)
x_train = d.iloc[:188318,:]

y_train = train["loss"]
x_test = d.iloc[188318:,:]
x_train.shape,x_test.shape,y_train.shape
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
f = {"id":test["id"],"loss":y_pred}

f = pd.DataFrame(f)

f.to_csv("submission.csv",index=False)
f.head()