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
data = pd.read_csv('../input/FRvideos.csv')
data.head()
list1= [data.views]
list2=[data.likes]
list3=[data.dislikes]
list4=[data.comment_count]
newdata = [data.views,data.likes,data.dislikes,data.comment_count]
print (list1 , '----------------',list2,'----------------',list3,'---------------',list4)

#Im going to find +300K views films and their values like like,dislike,comment
a = ['views','likes','dislikes','comment_count']
data = data[a]
data.head()
data.info() #i have to find miss info and remove them from data to analyze true data
# it seems like full i can show correlation 
x = data.corr()
#visualization
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x, annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
# i think data correlation table may help us to understand
x
# now i can show that likes,dislikes,comment count and their relations

sns.lmplot(x='likes', y='comment_count', data=data, size=8)
sns.lmplot(x='dislikes', y='comment_count', data=data, size=8)
plt.show()

sns.pairplot(data)
plt.show()
# i want to show top 5 most watched videos and their like, dislike and comment counts
data.insert(loc=0, column='sorted_views', value=sorted(data.views))
data.tail()
plt.figure(figsize=(15,10))
sns.barplot(x=data.sorted_views.loc[40719:40723], y=data.likes)
plt.xticks(rotation= 45)
plt.xlabel('Top 5 most watched video views in dataset')
plt.ylabel('Likes')

plt.figure(figsize=(15,10))
sns.barplot(x=data.sorted_views.loc[40719:40723], y=data.dislikes)
plt.xticks(rotation= 45)
plt.xlabel('Top 5 most watched video views in dataset')
plt.ylabel('Dislikes')
