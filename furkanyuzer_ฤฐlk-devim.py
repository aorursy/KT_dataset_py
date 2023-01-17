# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv")
data.info()
data.columns
#datadaki boşlukları siler
data.columns=[each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in data.columns]
data.count()
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.tail(10)
BTC=data["Volume_(BTC)"]
data.Close.plot(kind="line",color="r",label="btc close price",linewidth=1,alpha=0.5,grid=True,linestyle=":",figsize=(13,13))
BTC.plot(kind="line",color="blue",label="volume",linewidth=1,alpha=0.5,grid=True,linestyle=":")
plt.show()
data.plot(kind="scatter",x="Open",y="Close",color="blue",alpha=0.7)

plt.title("open ve closenin ikili karşılaştırması")
data.Weighted_Price.plot(kind="hist",bins=100,figsize=(14,14))
data.Weighted_Price.plot(kind="hist",bins=100,figsize=(14,14))
plt.clf()
print("Open" in data)
open_filtresi=data["Open"]>6060.00
close_filtresi=data["Close"]>6060.00
data[(open_filtresi)&(close_filtresi)]