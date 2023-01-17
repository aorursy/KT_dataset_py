# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/youtube-new/USvideos.csv')
data.head()
# Data frames from dictionary

country = ['Spain','France']

population = ['11','12']

list_label = ['country','population']

list_col = [country,population]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df['capital'] = ['Madrid','Paris']

df
# Broadcasting

df['income'] = 0 #broadcastion entire column

df
# Plotting all data 

data1 = data.loc[:,["views","likes","dislikes"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# Scatter plot

data1.plot(kind = "scatter",x="likes",y = "dislikes")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "likes",bins = 50,range= (0,250))
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "likes",bins = 50,range= (0,250),ax = axes[0])

data1.plot(kind = "hist",y = "likes",bins = 50,range= (0,250),ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # date is string

# we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# In order practice lets take head of youtube data and add it a time list

data2 = data.head()

date_list = ["1992-01-10", "1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

#lets make date as index

data2 = data2.set_index("date")

data2
# we will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not create from us like data2) we can solve this problem with interpolate

# We can interpolate from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")