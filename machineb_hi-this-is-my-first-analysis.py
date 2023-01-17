# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
data.info()
data.corr()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".2f",ax=ax)
plt.show()

data.head(10)
data.describe()
data.columns
data.Open.plot(kind="line",color="g",label="Open", linewidth=1,alpha=0.5,grid=True,linestyle=":")
data.Close.plot(color="r",label="Close", linewidth=1,alpha=0.5,grid=True,linestyle="-.")
plt.legend()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot for open n close prices")
plt.show()
data.plot(kind="scatter",x="Open",y="Close",alpha=0.5,color="red")
plt.xlabel("Open")
plt.ylabel("Close")
plt.title("Open Close Prices Combining")
data.High.plot(kind="hist",bins=10,figsize=(12,12))
plt.show()
data.High.plot(kind="hist",bins=10,figsize=(12,12))
plt.clf()
a=data["Open"]>29500
data[a]

data[(data["Open"]>650000) & (data["High"]>600000)]