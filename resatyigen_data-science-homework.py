# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/USvideos.csv") # import data set
print(data.columns) # get data columns
print(data.channel_title) # get channel titles
print(data.info())

# select random channel and show likes and dislikes status
# I select channel Vox

ch = data[data.channel_title == "Vox"]
# i use lineplot

#plt.plot(ch.publish_time,ch.likes,color = "blue", label = "likes",figsize = (18,18))
#plt.plot(ch.publish_time,ch.dislikes, color = "red", label = "dislikes",figsize = (18,18))

ch.likes.plot(kind = 'line', color = 'blue',label = 'Likes',linewidth=2,alpha = 0.7,grid = True,figsize = (18,18))
ch.dislikes.plot(color = 'red',label = 'Dislikes',linewidth=2,alpha = 0.7,grid = True,figsize = (18,18))

plt.legend() # show labels and set locations
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
ch["publish_time"] = pd.to_datetime(ch["publish_time"]) # convert to datetime

ch = ch.set_index("publish_time") # set a new index publish_time
ch.head()

ch = ch.loc[:,["views","likes","dislikes","comment_count"]]
ch.resample("A").sum() # a year
ch.resample("M").mean() # a month
ch.resample("D").mean().interpolate("linear") # Or we can interpolate with mean()

#print(ch.loc[:,["channel_title","video_id","likes","dislikes"]])

cl_data = data.loc[:,["channel_title","likes","dislikes","views"]] # relocated data frame


sm_data =  cl_data.groupby(['channel_title']).sum().reset_index() # group by channel_title,sum likes and dislikes, reset index for columns reindexing
#flt_data = sm_data[sm_data.likes>1000] # create filter 1000 likes

flt_data_likes = sm_data.sort_values(by=["likes"], ascending = False)
flt_data_dislikes = sm_data.sort_values(by=["dislikes"], ascending = False)
flt_data_views = sm_data.sort_values(by=["views"], ascending = False )

print(flt_data_likes.head(10))
print(flt_data_dislikes.head(10))
print(flt_data_likes.columns)
likes = flt_data_likes.head(10)
likes.plot(kind = "bar", x = "channel_title", y = "likes", color = "blue", figsize = (20,18))

#plt.bar(.channel_title,flt_data_likes.head(10).likes)
plt.title("Firs Most 10 Likes")
plt.xlabel("Channel Name")
plt.ylabel("Total Like")
plt.show()
dislikes = flt_data_dislikes.head(10)
dislikes.plot(kind = "bar", x = "channel_title", y = "dislikes", color = "red", figsize = (20,18))

#plt.bar(flt_data_dislikes.head(10).channel_title,flt_data_dislikes.head(10).dislikes, color = "red")
plt.title("Firs Most 10 Dislikes")
plt.xlabel("Channel Name")
plt.ylabel("Total Dislike")
plt.show()
views = flt_data_views.head(10)
views.plot(kind = "bar", x = "channel_title", y = "views", color = "teal", figsize = (20,18))

#plt.bar(flt_data_dislikes.head(10).channel_title,flt_data_dislikes.head(10).dislikes, color = "red")
plt.title("Firs Most 10 Views")
plt.xlabel("Channel Name")
plt.ylabel("Total View")
plt.show()
sm_data.describe() # ignore null entries, statics values