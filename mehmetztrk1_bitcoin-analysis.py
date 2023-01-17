# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")

data.head()

data.tail(5)

data.columns

data.info()
f,g=plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True, fmt='.1f',ax=g)

plt.show()
x=np.min(data["Close"])

data[data["Close"]==x].Open  #en düşük kapanışta bitcoin açılışı ne kadardı?



data2=data["Close"]-data["Open"] #açılış kapanış farkı

print("bir günde en çok ne kadar kazandırdı?",data2.max())

print("bir günde en çok ne kadar kaybettirdi?",-data2.min())

data2.plot(kind="line",color="green",label="Açılış-Kapanış Farkı",grid=True,linestyle="-",figsize=(15,15))

data.High.plot(kind="line",color="red",label="max değer",grid=True,linestyle="--",figsize=(15,15))

data.Low.plot(kind="line",color="blue",label="min değer",grid=True,linestyle=":",alpha=0.5,figsize=(15,15))

plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()





data.plot(kind="scatter",x="High",y="Low",color="Red",alpha=0.1)

plt.xlabel("High")

plt.ylabel("Low")

plt.show()


