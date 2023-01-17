# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
#first look to our dataset
data.head(10)
data.info()
#a new dataset is created to obtain necessary data only
data2= data[['title','views','likes','dislikes','comment_count']]
#first look to new dataset
data2.head(10)
#to see is there any null cells
data2.info()
#basic statistical info about new dataset
data2.describe()
#heatmap for correlations
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data2.corr(),annot = True,linewidths=.5,fmt = '.1f',ax=ax,cmap='YlGnBu')
plt.title('Heat Map')
plt.show()
#line plots for likes and dislikes for each channel
data2.likes.plot(kind = 'line',color = 'r',label = 'likes',lw = 1,alpha = 0.5, grid = True, figsize=(12,12))
data2.dislikes.plot(kind = 'line',color = 'g',label = 'dislikes',lw = 1,alpha = 0.5, grid = True)
#scatter plots for likes vs views and dislikes vs views
data2.plot(kind='scatter',x='views',y='likes',color='r',label='likes',alpha=0.5,grid=True,figsize=(12,12))
data2.plot(kind='scatter',x='views',y='dislikes',color='g',label='dislikes',alpha=0.7,grid=True,figsize=(12,12))
plt.legend()
plt.show()

#shows each channels' trending video numbers
data3 = data.groupby('channel_title')['video_id'].nunique()
print(data3)
#channels' trending video numbers Histogram
data3.plot(kind = 'hist',bins = 100,figsize = (16,9))
plt.show()
#lambda function that calculating activity rates 
i=0
for i in data2.index:
    activity_rate = lambda x,i: (x[i]/data2.views[i])*100
    
    data2['like_rates'][i]=activity_rate(data2.likes,i) #create reated columns in the dataset
    data2['dislike_rates'][i]=activity_rate(data2.dislikes,i)
    data2['comment_rates'][i]=activity_rate(data2.comment_count,i)
data2.columns
data2.info()
print("like rates(max,min):",max(data2.like_rates),
min(data2.like_rates))
print("dislike rates(max,min):",max(data2.dislike_rates),
min(data2.dislike_rates))
print("comment rates(max,min):",max(data2.comment_rates),
min(data2.comment_rates))
