import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import re

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/used-iphone-11-prices-in-us/filename.csv")

print(df.shape)

df.head()
df = df[df.GB != "0"]

df = df[df.Color != "0"]
for index,row in df.iterrows():

    color = row['Color']

    gb = row['GB']

    df.at[index,'Description'] = "iPhone 11"

    df.at[index,'Price'] = round(row['Price'])

df.GB = df.GB.str.replace('GB', '').astype(int)

df.Price = df.Price.astype(int)

df.head()
color = df.Color.unique()

color.sort()

color
df["Color"].value_counts().plot(kind="barh")
color = df.Color.unique()

color.sort()

col_dict = dict(zip(color, range(len(color))))

df.Color.replace(col_dict, inplace=True)

print(col_dict)
df.head()
df["Pro?"].value_counts().plot(kind="barh")
df["Unlock?"].value_counts().plot(kind="barh")
df["Max?"].value_counts().plot(kind="barh")
df["GB"].value_counts().plot(kind="barh")
df.Price.hist()
y = df.Price

x = df.drop(columns=["Description", "Price"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn import linear_model



clf = linear_model.ARDRegression()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

accuracy = clf.score(x_test, y_test)

print(accuracy*100,'%')