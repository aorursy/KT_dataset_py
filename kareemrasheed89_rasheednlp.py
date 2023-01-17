# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CA=pd.read_json('../input/youtube-new/CA_category_id.json')

CA.head(5)
CAV=pd.read_csv('../input/youtube-new/CAvideos.csv',parse_dates=['trending_date','publish_time'])

CAV.head(5)
CAV.info()
print("description_null")

print("-"*10)

print(CAV['description'].isnull().sum())
CAV.describe()
CAV.tail(3)
CAV=pd.DataFrame(CAV)

CAV.head(3)
CAV.info()

CAV.shape
CAV['video_id']=CAV['video_id'].astype(str)
CAV['description'].value_counts()
CAV['video_id'].value_counts()
print(CAV['trending_date'],['publish_time'])

import datetime as dt

CAV['publish_months']=CAV['publish_time'].dt.year*100+CAV['publish_time'].dt.month

CAV['publish_months']

CAV['publish_time']

CAV['trending_date']
CAV.info()
CAV['publish_months'].astype(float).astype(str)

CAV['trending_date']=CAV['trending_date'].str.replace('.',"")
CAV['trending_date']

CAV.head(2)
Likes=CAV['likes'].sum()

views=CAV['views'].sum()

Dislikes=CAV['dislikes'].sum()

Comments=CAV['comment_count'].sum()
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(10,5))

plt.plot(CAV[0:12]["channel_title"],CAV[0:12]["views"]/1000,c='red')

plt.plot(CAV[12:24]["channel_title"],CAV[12:24]["views"]/1000,c='blue')

plt.xticks(rotation=90)

plt.xlabel("Channel Title")

plt.ylabel("Views")

plt.show()
print("Canadian Videos Summary")

print("Likes       Views     Dislikes      Comments")

print("-"*45)

print(Likes,  views,  Dislikes,    Comments )

fig = plt.figure(figsize=(6,15))

ax1 = fig.add_subplot(4,1,1)

ax2 = fig.add_subplot(4,1,2)

ax3 = fig.add_subplot(4,1,3)

ax4 = fig.add_subplot(4,1,4)

ax1.hist(CAV['views']/1000,bins=20,range=(0,5))

ax1.set_ylim(0,100)

ax1.set_title("views['1000]")

ax2.hist(CAV['comment_count']/1000,bins=20,range=(0,5))

ax2.set_ylim(0,10000)

ax2.set_title('Comments["1000]')

ax3.hist(CAV['likes']/1000,bins=20,range=(0,5))

ax3.set_ylim(0,10000)

ax3.set_title("Likes['1000]")

ax4.hist(CAV['dislikes']/1000,bins=20,range=(0,5))

ax4.set_ylim(0,10000)

ax4.set_title("Dislikes['1000]")

plt.show()
fig=plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(4,1,1)

ax2 = fig.add_subplot(4,1,2)

ax3 = fig.add_subplot(4,1,3)

ax4 = fig.add_subplot(4,1,4)

ax1.plot(CAV['publish_time'],CAV["views"]/1000,c="blue")

ax1.set_title("Views By Publish Time")

ax2.plot(CAV['publish_time'],CAV["likes"]/1000,c="green")

ax2.set_title("Likes By Publish Time")

ax3.plot(CAV['publish_time'],CAV["dislikes"]/1000,c="red")

ax3.set_title('Dislikes By Publish Time')

ax4.plot(CAV['publish_time'],CAV["comment_count"]/1000,c="maroon")

ax4.set_title("Comment_Counts By Publish Time")

plt.show()
fig=plt.figure(figsize=(15,20))

ax1 = fig.add_subplot(4,1,1)

ax2 = fig.add_subplot(4,1,2)

ax3 = fig.add_subplot(4,1,3)

ax4 = fig.add_subplot(4,1,4)

ax1.plot(CAV[0:10]['channel_title'],CAV[0:10]["views"]/1000,c="blue")

ax1.set_title("Views By Channel Title")

ax2.plot(CAV[0:10]['channel_title'],CAV[0:10]["likes"]/1000,c="green")

ax2.set_title("Likes By Channel_=Title")

ax3.plot(CAV[0:10]['channel_title'],CAV[0:10]["dislikes"]/1000,c="red")

ax3.set_title('Dislikes By Channel Title')

ax4.plot(CAV[0:10]['channel_title'],CAV[0:10]["comment_count"]/1000,c="maroon")

ax4.set_title("Comment_Counts By Channel Title")

plt.show()

CAVgroup=CAV.groupby('channel_title')

By_Views=CAV.sort_values(["views","channel_title"],ascending=False).head(10)

By_Likes=CAV.sort_values(["likes","channel_title"],ascending=False).head(10)

By_Dislikes=CAV.sort_values(["dislikes","channel_title"],ascending=False).head(10)

By_Comments=CAV.sort_values(["comment_count","channel_title"],ascending=False).head(10)

By_Dislikes
fig=plt.figure(figsize=(15,20))

ax1 = fig.add_subplot(4,1,1)

ax2 = fig.add_subplot(4,1,2)

ax3 = fig.add_subplot(4,1,3)

ax4 = fig.add_subplot(4,1,4)

ax1.plot(By_Views['channel_title'],By_Views['views']/1000,c="blue")

ax1.set_title("Views By Channel Title")

ax2.plot(By_Likes["channel_title"],By_Likes["likes"]/1000,c="green")

ax2.set_title("Likes By Channel_Title")

ax3.plot(By_Dislikes["channel_title"],By_Dislikes["dislikes"]/1000,c="red")

ax3.set_title('Dislikes By Channel Title')

ax4.plot(By_Comments["channel_title"],By_Comments["comment_count"]/1000,c="maroon")

ax4.set_title("Comment_Counts By Channel Title")

plt.show()
CAVHV=CAV['views']>1000000

CAV[CAVHV].plot(kind="scatter",x="comment_count",y="views")

plt.show()