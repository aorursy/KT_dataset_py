# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/FRvideos.csv")
data.info()

# We can see what is in the data by using this command. This data includes

# video_id/trending_date/title/channel_title
data.corr()

# correlation map

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot=True, linewidth=.8, fmt=".2f",ax = ax)

plt.show()
data.head()
data.columns
# Line Plot

data.likes.plot(kind = "line",color ="g", label ="Likes",grid =True, alpha=0.5, linestyle = ":")

data.dislikes.plot(kind = "line", color ="r", label="Dislikes",linestyle="-.",alpha=0.5)

plt.legend(loc="upper right")

plt.xlabel("videos")

plt.title("Likes and Dislikes of videos on Youtube")

plt.show()
# scatter plot x=likes, y=dislikes

data.plot(kind="scatter", x="likes", y="dislikes",color="red")

plt.title("Likes vs Dislikes")

plt.show()
# histogram

data.likes.plot(kind="hist",bins=60, xlim=[0,max(data.likes)], ylim=[0,50])

plt.show()
series = data['likes']

print(type(series))

data_frame=data[["likes"]]

print(type(data_frame))
data[data["likes"]> max(data["likes"])/2]
data[data["dislikes"]> max(data["dislikes"])/2]
data[(data["views"]> max(data["views"])/2) & (data["likes"]> max(data["likes"])/2) ]
for index,value in data[["likes"]][0:5].iterrows():

    print(index,":", value)

data.info()

data.shape

data.describe()
print(data.dislikes.value_counts())
data.boxplot(column='likes')
data_new=data.head(15)
data_new.boxplot(column='likes')
data_melt = pd.melt(frame=data_new, id_vars='video_id',value_vars=['views','likes','dislikes'])
data_melt.pivot(index='video_id',columns='variable',values='value')