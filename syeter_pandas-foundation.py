# This Pytho =n 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/NFA 2018.csv')
footballer = ["Ronaldinho","Sabri"]
position = ["midfield","wholefield"]
list_label = ["footballer","position"]
list_col = [footballer,position]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Add new columns
df ["Special Ability"] = ["Tackling with more than one players","Crazy shoots from anywhere in the field "]
df
# Broadcasting
df["income"] = 0
df
# Plotting all data 
data1 = data.loc[:,["crop_land","grazing_land","built_up_land"]]
data1.plot()
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot
data1.plot(kind ="scatter",x="crop_land",y = "grazing_land")
plt.show()
# hist plot
data1.plot(kind = "hist",y = "grazing_land",bins = 50,range = (0,250),normed = True)
# histogram subplot with non cumulative and cumulative
fig,axes = plt.subplots(nrows = 2,ncols = 1)
data1.plot(kind = "hist",y = "crop_land",bins = 50,range = (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "crop_land",bins = 50,range = (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe() # only numeric variable would be shown
time_list = ["1996-10-30","1996-5-12"]
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")

data2 = data.head()
date_list = ["1992-01-10","1992-03-10","1992-05-10","1992-07-15","1992-11-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object

data2 = data2.set_index("date")
data2

print(data2.loc["1992-03-10"])
print(data2.loc["1992-03-10":"1992-11-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
