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
data = pd.read_csv("/kaggle/input/youtube-new/GBvideos.csv")

data.head()
#veri listeleri hazırlanır

title = ["Reamonn - Supergirl", "Eagles - Hotel California"]

views = ["12020","52874"]



#label ve col değerleri listeye eklenir

list_label=["title","views"]

list_colm =[title,views]



# convert zip

df_zip = list(zip(list_label,list_colm))



# convert dict

df_dict = dict(df_zip)



#convert df

df=pd.DataFrame(df_dict)

df
# yeni column ekleme

df["likes"]=["12000","35698"]

df
# broadcasting : yeni column ekleme ve tüm elemanlara aynı değeri atama

df["dislikes"]=0

df
# Visual Exploratory Data Analysis



# Plot



data_nw = data.loc[:20,["likes","dislikes"]]

data_nw.plot()
# Subplot



data_nw.plot(subplots=True, color=["orange","green"])
# Scatter



data_nw.plot(kind="scatter",x="likes",y="dislikes")

plt.show()
# Histogram



data_nw.plot(kind="hist", y="likes", bins=30, color="purple", range=(0,100000), normed=True )
# Histogram Cumulative



data_nw.plot(kind="hist", y="likes", bins=30, color="purple", alpha=.3, range=(0,100000), normed=True, cumulative=True )
# Statistical Exploratory Data Analysis



data.describe()
# Indexing Pandas Time Series



time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1]))
# convert to date time object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]



datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

data2= data2.set_index("date")

data2
# index date time old göre date time'a göre verileri çekebiliriz

print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
# A - yıllar

data2.resample("A").mean()
# M - Aylar

data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")