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
data = pd.read_csv('../input/data.csv')
data.info()
data.head()
data.tail()
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



#terms



video_Uploads = normalize_df1["Video Uploads"]

video_Views = normalize_df1["Video views"]



#graph



video_Views.plot(kind = "line", color = "r",label = "Video Views",linewidth = 2,grid = True,linestyle = ":")

normalize_df1.Subscribers.plot(kind = "line", color = "b",label = "Subscribers",linewidth = 2,grid = True,linestyle = "--")

plt.legend()

plt.title("Line Plot")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
normalize_df1.plot(kind='scatter', x='Subscribers', y='Video views',alpha = 0.5,color = 'blue')

plt.title("Scatter Plot")

plt.show()
top500 = normalize_df1.head(500)

top500.corr()
video_Views1 = top500["Video views"]



top500.Subscribers.plot(kind = "line",label = "Video Views", color = "r", linewidth = 1,grid = True,alpha =0.5 )

video_Views1.plot(kind = "line",label = "Subscribers", color = "b", linewidth = 1,grid = True,alpha =0.5 )

plt.legend()

plt.title("Line Plot")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')

df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')
df.info()
sns.jointplot(data=df, x='Subscribers', y='Video views')

plt.show()
top_500df = df[0:500]

sns.lmplot(data=top_500df,x='Video views', y='Subscribers', hue='Grade', palette='Set1')

plt.title('Top 500 Rank Channels')

plt.show()
df.corr()
sns.heatmap(df.corr(), cmap='viridis')

plt.tight_layout()
plt.subplots()

sns.regplot(x=df['Video views'], y=df["Subscribers"], fit_reg=True,scatter_kws={"color":"red"})

plt.show()
plt.subplots()

sns.regplot(x=df['Video views'], y=df["Video Uploads"], fit_reg=True,scatter_kws={"color":"red"})

plt.show()
plt.subplots()

sns.regplot(x=df['Video Uploads'], y=df["Subscribers"], fit_reg=True,scatter_kws={"color":"red"})

plt.show()
sns.lmplot(x='Video views', y='Subscribers', data=df, fit_reg=True, hue='Grade')

plt.show()
df.sort_values(by = ['Video views'], ascending = True).head(9).plot.barh(x = 'Channel name', y = 'Subscribers')

plt.show()
#End of this test