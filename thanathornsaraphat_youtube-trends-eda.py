# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/youtube-new/USvideos.csv")
data.head()
data.columns
data.describe()
data.info()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.drop(["video_id", "trending_date", "category_id", "tags", "thumbnail_link", "comments_disabled", "ratings_disabled", "video_error_or_removed", "description"],axis=1,inplace=True)
data.info()
data.head()
data.describe()
filter1 = data.likes >= 1000000
data.likes.idxmax()
data.loc[38273]
filter2 = data.publish_time > "2018-01-01T15:00:02.000Z" 
#data.loc[:, "channel_title"]
data[filter1 & filter2]
channel = data[data.channel_title == "ibighit"]
channel2 = data[data.channel_title == "Logan Paul Vlogs"]
plt.figure(figsize=(15,10))
plt.plot(channel.publish_time, channel.likes, color = "blue", label = "ibighit")
plt.plot(channel2.publish_time, channel2.likes, color = "green", label = "Logan Paul Vlogs")
plt.xticks(rotation= 90)
plt.xlabel("Publish Time")
plt.ylabel("Like Counts")
plt.legend()
plt.show()
a = data.channel_title[:20]
b = data.likes[:20]
plt.figure(figsize=(15,10))
sns.barplot(x= a, y=b)
plt.xticks(rotation= 90) # put the labels in degree of 45
plt.xlabel('Channel Title')
plt.ylabel('Like')
plt.title('Like of Channel')
# Line Plot
# kind = type of plot(barplot, scatter plot, line plot vs.), alpha = opacity
channel.likes.plot(kind = 'line', color = 'g',label = 'Like Counts',linewidth=1,alpha = 0.8,grid = True,linestyle = 'solid') # b = blue, g = green, r = red, c = cyan, m = magenta, y = yellow, k = black, w = white in color abbreviations
channel.dislikes.plot(color = 'r',label = 'Dislike Counts',linewidth=1, alpha = 0.8,grid = True,linestyle = 'solid')
plt.legend(loc='upper left') # legend = puts label into plot
plt.xlabel('x-axis')    # label = name of label
plt.ylabel('y-axis')
plt.title('Line Plot of Likes & Dislikes of ibighit') # title = title of plot

plt.show()
# Scatter Plot 
# x = likes, y = views
plt.scatter(channel.likes, channel.views, color = "red", label = "ibighit", alpha = 0.5)
plt.scatter(channel2.likes, channel2.views, color = "blue", label = "ibighit", alpha = 0.5)
plt.xlabel("Like Counts")
plt.ylabel("View Counts")
plt.title("Like & View Counts of Scatter Plot")
plt.legend()
plt.show()
channel.info()
plt.hist(channel.views, bins = "auto", color = "#86bf91", alpha = 1.0, rwidth=0.85)  # bin = bar sayisi
plt.xlabel("View Count")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.figure(figsize=[10,18])
fig = plt.figure(figsize = (15,8))
ax = fig.gca()
channel.hist(ax=ax, bins = "auto", color = "#86bf91", grid = False, rwidth=0.85)
plt.show()
