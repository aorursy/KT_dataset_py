# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from datetime import datetime as dt 

import seaborn as sns 

import matplotlib.pyplot as plt
string_date = ["12122013","12122012","27021995","17051990","04042015","30092015","04092019"]
for date in string_date:

    print(dt.strptime(date,"%d%m%Y"))
data = pd.read_csv("../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")
data.index = data["Date"]
data.head()
data.Open.plot(kind = "line", alpha = .5, color = "purple")

plt.xlabel("dates")

plt.ylabel("open prices")

plt.title("tesla ")

plt.show()
data.info()
corelation = data.corr()
f,axis = plt.subplots(figsize=(15,15))

sns.heatmap(corelation, annot = True,linewidths = .5,fmt = ".1f",ax = axis)

plt.show()
from datetime import datetime as dt

from pytz import timezone as tz
data["Open"].rolling(window = 3).mean()
data.head()

#data.drop("date",axis=1,inplace = True)
data.columns = [each.lower()  for each in data.columns]
#data.resample("3W") does not working but ı dont know :) 
correlation = data.rolling(window = 2).mean().dropna().head(25)

x = correlation.corr()
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(x,annot = True,linewidths = .5,fmt = ".1f",ax = ax)

plt.show()

#correlation explain relationship beetween two data 
#data.drop(["Date"],axis = 1,inplace = True)

data.columns = [col.lower() for col in data.columns]
data.plot(kind = "line",color = "purple", y = 'volume',alpha = 0.5)

plt.xlabel("date")

plt.ylabel("volume")

plt.title("tesla data")

plt.xticks(rotation = 45)

plt.show()
data.close.plot(color = "purple",alpha = .9,style = ".",legend = True)

data["adj close"].plot(color = "red",alpha = .1,style = ".",legend = True)

plt.ylim((15,280))

plt.xticks(rotation = 45)