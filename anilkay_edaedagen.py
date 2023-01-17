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
us=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

us.head()
print(len(set(us["channel_title"])))

print(len(us["channel_title"]))
read=us["channel_title"].groupby(us["channel_title"]).count()

print(read)
read.idxmax()
read.sort_values(ascending=False)[0:20]
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
sns.lmplot(data=us,x="likes",y="views")
sns.lmplot(data=us,x="views",y="likes")
sns.lmplot(data=us,x="views",y="dislikes")
sns.relplot(data=us,x="views",y="dislikes")
removed=us[us["video_error_or_removed"]==True]

removed.head()
sns.countplot(data=removed,x="comments_disabled")
set(removed["channel_title"])
removed[removed["channel_title"]=="googledoodles"]["description"]
sns.relplot(data=us,x="views",y="likes",hue="comments_disabled")
sns.relplot(data=us,x="views",y="likes",hue="ratings_disabled")
categories=pd.read_json("/kaggle/input/youtube-new/US_category_id.json")

categories.head()
categories["items"][0]
categories["items"][0]["snippet"]["title"]
kategoriler=dict()

for cat in categories["items"]:

    title=cat["snippet"]["title"]

    catid=cat["id"]

    kategoriler.update({title: catid})

kategoriler
kategoriler2=dict()

for cat in categories["items"]:

    title=cat["snippet"]["title"]

    catid=cat["id"]

    kategoriler2.update({catid:title})

kategoriler2
original=us["category_id"]
values=list()

for o in original:

    value=kategoriler2[str(o)]

    values.append(value)

values    
us["category_id"]=values
us.head()
plt.figure(figsize=(30, 15))

sns.countplot(data=us,x="category_id")
cat_views=us["views"].groupby(us["category_id"]).sum()

cat_views
cat_views.sort_values(ascending=False)
cat_views_mean=us["views"].groupby(us["category_id"]).mean()

cat_views_mean.sort_values(ascending=False)
cat_views_max=us["views"].groupby(us["category_id"]).max()

cat_views_max.sort_values(ascending=False)