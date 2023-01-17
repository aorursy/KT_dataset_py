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
data = pd.read_csv("../input/efw_cc.csv")
data.head()
#data.tail()
data.columns
data.info()
print(data['1_size_government'].value_counts(dropna = False))
data.plot(kind='scatter', x='1a_government_consumption', y='1_size_government',alpha= 0.5,color = 'red')
plt.xlabel('consumption')
plt.ylabel('size_government')
plt.show()

data.boxplot(column='1a_government_consumption', by='1_size_government')
data_new = data.head()
melted = pd.melt(frame=data_new,id_vars= 'countries', value_vars= ['rank','1a_government_consumption'])
melted
melted.pivot(index = 'countries', columns='variable', values='value')

data1 = data['countries'][:10]
data2 = data['rank'][:10]
conc_data_col = pd.concat([data1,data2], axis = 1)
conc_data_col
data.dtypes
data['rank'] = data['rank'].astype('float')
data.dtypes
data.info()
data["rank"].value_counts(dropna= False)
data3 = data
data3["rank"].dropna(inplace = True)
data3["rank"].value_counts(dropna= False)
assert data["rank"].notnull().all()
data["rank"].fillna('empty', inplace = True)
assert data["rank"].notnull().all()
school = ["YTU", "ODTU", "EGE"]
city = ["istanbul", "Ankara", "Izmir"]
list_label = ["school","city"]
list_col = [school,city]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df 
df["number_of_student"] = [5000, 6000, 2000]
df
df["departmen"] = " "
df
data.head()
data1 = data.loc[:,["year","rank"]]
data1.plot()
data1.plot(subplots=True)
plt.show()

data1.plot(kind="scatter", x = "year", y= "rank")
plt.show()
data1.plot(kind="hist", y="rank", bins= 50, range= (0,160),normed = True)
data.describe()
data2 = data.head()
data_time = ["1992","1993","1994","1995","1996"]
data_time_object = pd.to_datetime(data_time)
data2["date"] = data_time_object
data2 = data2.set_index("date")
data2
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").interpolate("linear")
data = pd.read_csv("../input/efw_cc.csv")
dictionary = {"name" : ["Mehmet", "Ali", "Ayse"], "city" : ["Bursa", "Istanbul","Ankara"]}
df = pd.DataFrame(dictionary)
print(type(df))
df
df = df.set_index("#")  # I couldn't add # for index 
df
print(type(df["name"]))
print(type(df[["name"]]))
df.loc[:1:-1,"name"]

data.head()
data.info()
data.rank.asty
filter1 = data.rank > 150
filter2 = data.year > 2015
data[filter1 & filter2]

