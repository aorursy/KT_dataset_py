#I show distinct plots where we can see the relationship between features and how it works the Rank made by 
#socialblade

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the "../input/" directory.

import os
df = pd.read_csv("../input/data.csv")

df.head()
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')
df.info()
sns.jointplot(data=df, x='Subscribers', y='Video views')
top_100df = df[0:100]
sns.jointplot(data=df, y='Subscribers', x='Video Uploads')
sns.jointplot(data=df, x='Video Uploads', y='Video views')
sns.lmplot(data=top_100df,x='Subscribers', y='Video views', hue='Grade', palette='Set1')
plt.title('Top 100 Rank Channels')
df.corr()
sns.heatmap(df.corr(), cmap='viridis')
plt.tight_layout()
