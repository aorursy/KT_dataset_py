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
data = pd.read_csv("../input/marvel-wikia-data.csv")
# we create dataframe with pd.read_csv code.
data.columns

#Also you can try this code for create dataframe
eye = ["Black","Blue"]
age = ["25","46"]
list_label = ["eye","age"]
list_column = [eye,age]
zipped = dict(zip(list_label,list_column))
df = pd.DataFrame(zipped)
df
#you can add new columns
df["gender"] = ["female","male"]
df
#broadcasting
df["income"] = 0
df
data1 = data.loc[:,["page_id","APPEARANCES"]]
data1.plot()
data1.plot(subplots = True)
plt.show()
#scatter plot
data1.plot(kind="scatter",x="APPEARANCES",y="page_id")
plt.show()
#histogram plot
data1.plot(kind="hist",y ="APPEARANCES",bins=50,range=(0,250),normed = True)
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind="hist",y = "APPEARANCES",bins=50, range=(0,250),normed=True,ax = axes[0])
data1.plot(kind="hist",y = "APPEARANCES",bins=50, range=(0,250), normed = True, ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ["1995-05-05","1997-07-12","1990-07-05"]
print(type(time_list[1]))
#this is a string
#you may want to create datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
data2 = data.head()
data2
date_list = ["1950-01-10","1965-12-12","1992-03-10","1998-01-11","1963-03-15"]
datetime_object = pd.to_datetime(date_list)
data2["date"]=datetime_object
data2 = data2.set_index("date")
data2
print(data2.loc["1950-01-10"])
print(data2.loc["1950-01-10":"1963-03-15"])
data2.resample("A").mean()
#A=year
#M = month
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")

data2.resample("M").mean().interpolate("linear")




