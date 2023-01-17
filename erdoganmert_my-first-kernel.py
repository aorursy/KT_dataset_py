# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv")

print(df.info())
df["Video Uploads"] = pd.to_numeric(df["Video Uploads"],errors='coerce')

df["Subscribers"] = pd.to_numeric(df["Subscribers"],errors='coerce')
df.info()
df.corr()
# correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df1 = df.drop(df.loc[:,:"Channel name"],axis = 1)
normalize_df1 = (df1-df1.min())/(df1.max()-df1.min())
video_Uploads = normalize_df1["Video Uploads"]

video_Views = normalize_df1["Video views"]

video_Views.plot(kind = "line", color = "g",label = "Video Views",linewidth = 1,grid = True,linestyle = "--")

normalize_df1.Subscribers.plot(kind = "line", color = "r",label = "Subscribers",linewidth = 1,grid = True,linestyle = ":")

plt.legend()

plt.title("Line Plot")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()

normalize_df1.plot(kind='scatter', x='Subscribers', y='Video views',alpha = 0.5,color = 'red')

plt.title("Scatter Plot")

plt.show()
top500 = normalize_df1.head(500)

top500.corr()


video_Views1 = top500["Video views"]



top500.Subscribers.plot(kind = "line",label = "Subscribers", color = "r", linewidth = 1,grid = True,alpha =0.5 )

video_Views1.plot(kind = "line",label = "Video Views", color = "g", linewidth = 1,grid = True,alpha =0.5 )

plt.legend()

plt.title("Line Plot")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
top500.plot(kind='scatter', x='Subscribers', y='Video views',alpha = 0.5,color = 'red')

plt.title("Scatter Plot")

plt.show()