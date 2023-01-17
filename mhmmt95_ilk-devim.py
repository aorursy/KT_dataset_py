# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv")
data
f,ax =plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True ,linewidths=.5,fmt=".1f",ax=ax)
plt.show()
data.tail(10)
data.columns
volume=data["Volume_(BTC)"]
data.Close.plot(kind="line",color="r",label="btc close price",linewidth=1,alpha=0.5,grid=True,linestyle=":")
volume.plot(kind="line",color="blue",label="volume",linewidth=1,alpha=0.5,grid=True,linestyle=":")
plt.show()
data.plot(kind="scatter",x="Open",y="Close",alpha=0.5,color="red")
plt.xlabel("Open")
plt.ylabel("Close")
plt.title("Open Close scatter plot")
plt.show()
data.Close.plot(kind="hist",bins=50,figsize=(12,12))
plt.show()
data.Close.plot(kind="hist",bins=50,figsize=(12,12))
plt.clf()
for key,value in data.items():
    print(key,":",value)
print("")