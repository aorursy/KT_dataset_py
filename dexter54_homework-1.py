# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#I will work on daily trending youtube videos of US by using given path above
#First ı will import some visualization libaries

import matplotlib.pyplot as plt

import seaborn as sns
#Now, I will read csv file that includes data about US trend videos by using given code above

trend_US=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
# I will use .info to get information about type of entries

trend_US.info()
# I used head to get information about first 5 rows

trend_US.head()
#Also, we can get information about more columns by giving int to head function

trend_US.head(10)
# You can call .columns to get information about features of trend videos

trend_US.columns
# I will call .corr to see correlation between features

trend_US.corr()

# We can reach some result by looking this table. For example, there are positive correlations between likes and views, likes and comment_count which means that if one increases, other will also increase

#I will use heatmap to see correlation clearly

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(trend_US.corr(),annot=True,linecolor="pink",vmin=-1.0,vmax=1.0,linewidths=.5,ax=ax,cmap="coolwarm",fmt=".1f")
#We can see correlation between views and likes by looking this scatter plot

plt.scatter(data=trend_US,x="views",y="likes",color="purple",alpha=0.5)

plt.xlabel("Views")

plt.ylabel("Likes")

plt.show()
#By using line plot, we are comparing likes and dislike of an video

trend_US["likes"].plot(kind="line",color="green",label="likes",grid=True,linewidth=1,alpha=0.5,linestyle="-")

trend_US["dislikes"].plot(kind="line",color="blue",label="dislikes",grid=True,linewidth=1,alpha=0.5,linestyle=":")

plt.legend(loc="upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Views and Comments")

plt.show()

#We can see that likes are mostly bigger than dislikes in trend videos

#By using histogram we can see the distribution of features



trend_US["comments_disabled"].astype("int").plot(kind="hist",bins=30) #I used astype to conver bool to integer

plt.show()

#We can conclude that  comments of almost all videos in US trends are open.

trend_US["ratings_disabled"].astype("int").plot(kind="hist",bins=30)

plt.show()

# Also almost all trend videos have rating system.
#I will try to find videos which are disabled comments

trend_US[trend_US["comments_disabled"]==True]

#I wonder is there any video which is both comment disabled and rating disabled

trend_US[(trend_US["comments_disabled"]==True)&(trend_US["ratings_disabled"]==True)]

# Wow, there are lots of videos
#I wonder is trend includes videos whose views less than 10000

trend_US[trend_US["views"]<10000]

# There are some videos with views less than even 1000 so entering trend does not depend on views