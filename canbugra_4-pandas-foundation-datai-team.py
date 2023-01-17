# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df

list1 = ["ali","veli","kenan","hilal","ayse","evren"]

list2 = [15,16,17,33,45,66]

list3 = [100,150,240,350,110,220]

list_label = ["Name","Age","Maas"]

list_col = [list1,list2,list3]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df

df["jobs"]=["worker","engineer","manager","officer","assistant","secretary"]

df
df["income"] = 0

df

data = pd.read_csv('../input/Pokemon.csv')

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

#subplots

data1.plot(subplots = True)

plt.show()
#scatter plot

data1.plot(kind="scatter",x="Attack",y="Defense")

plt.show()
#histogram

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True)

plt.show()
fig,axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt

#Indexing Pandas Time Series

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")



data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])

data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")