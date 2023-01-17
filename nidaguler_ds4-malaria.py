

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/malaria-dataset/reported_numbers.csv")
df.head()
df.columns=["country","year","case","death","who_region"]
df.head()
df.tail()
df.describe()
df.info()
df.isna().sum()
df=df.dropna()
df.isna().sum()
df.case.unique()
df[df["case"]==0.0]
#visualize the correlation

plt.figure(figsize=(15,10))

sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")

plt.show()