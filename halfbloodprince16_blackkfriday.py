import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/data"))
df = pd.read_csv("../input/data/train.csv")
df.head(2)
df['User_ID'] = df['User_ID']%1000000
cnt =[]

cnt = df.groupby(df.User_ID).count().head(5)
df.groupby(['User_ID','Product_ID']).count().head(10)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()

ohe = OneHotEncoder(categorical_features = [2])
df["Product_ID"] = le.fit_transform(df["Product_ID"])
df.head(2)
df["Marital_Status"].nunique()
gen = pd.get_dummies(df.iloc[:,[2,3,5,6]])
gen.head(2)
gen = pd.concat((df.iloc[:,[0,1,4,7]],gen),axis=1)
gen.head()
df.head(10)
df['Product_Category_2'] = df['Product_Category_2'].bfill()

df["Product_Category_3"] = df["Product_Category_3"].bfill()
df.head(1)
data = pd.concat((gen,df.iloc[:,[8,9,10,11]]),axis=1)
data.head()
data.to_csv("data.csv",index=False)