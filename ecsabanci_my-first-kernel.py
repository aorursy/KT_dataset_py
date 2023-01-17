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
data = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")



data.head()
#we should fill NA values by zero.

data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(value=0, inplace=True)

data['Open'].fillna(value=0, inplace=True)

data['High'].fillna(value=0, inplace=True)

data['Low'].fillna(value=0, inplace=True)

data['Close'].fillna(value=0, inplace=True)
data["average_btwn_OC"] = (data.Open)/(data.Close)
data['average_btwn_OC'].fillna(value=0, inplace=True)
data.High.max()
data.tail(15)
# I tried to observe last 15 datas!!



plt.figure(figsize=(20,7))



data.Open[3997682:3997695].plot(kind="line",color="blue",alpha=1,label="Opening Value",linewidth=1,linestyle="--",grid=True)

data.Close[3997682:3997695].plot(kind="line",color="red",alpha=1,label="Closing Value",linewidth=1,linestyle="-.",grid=True)



plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Opening Values vs Closing Values of BTC")



plt.legend()

plt.show()

# because of the openin and closing values are almost same I couldn't observe in the graph!!
plt.figure(figsize=(20,7))



data["Volume_(BTC)"][3997682:3997695].plot(kind="line",color="blue",alpha=1,label="Volume_(BTC)",linewidth=1,linestyle=":",grid=True)

data["average_btwn_OC"][3997682:3997695].plot(color="red",alpha=1,label="average_btwn_OC",linewidth=1,linestyle="-.",grid=True)

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Opening Values vs Closing Values of BTC")

plt.legend()

plt.show()
plt.scatter(data.Open[3997682:3997695],data.Close[3997682:3997695],color="red",alpha=.5)

plt.scatter(data["Volume_(BTC)"][3997682:3997695],data["average_btwn_OC"][3997682:3997695],color="blue",alpha=.5)
#increase of btc with the time

plt.scatter(data.Timestamp[3997682:3997695],data.High[3997682:3997695],color="red",alpha=.5)
data.High.plot(kind="hist",color="brown",bins=50,figsize=(15,15))
data.High.value_counts()