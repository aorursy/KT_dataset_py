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
import pandas as pd

import numpy as np

import json

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from datetime import time

import glob

%matplotlib inline
ca = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')
ca.head()
with open('/kaggle/input/youtube-new/US_category_id.json') as f:

    categories = json.load(f)['items']
category_name = {}

for category in categories:

    category_name[int(category['id'])] = category['snippet']['title']

ca['category_name'] = ca['category_id'].map(category_name)
print(category_name)
ca.info()
ca.isnull().sum()

# we see some nulls in the description and category name, let's find out why

# re ran to double check
ca['description'].fillna(value='None',inplace=True)
# converted object to dt obj

ca['publish_time'] = pd.to_datetime(ca['publish_time'], format = '%Y-%m-%dT%H:%M:%S.%fZ')
# getting months the video was published in

ca['publish_month'] = ca['publish_time'].dt.month
# trending date -> datetime

ca['trending_date'] = pd.to_datetime(ca['trending_date'],format = '%y.%d.%m')

ca['trending_month'] = ca['trending_date'].dt.month
ca.columns
# lets see how many videos are in each category

ca['category_name'].value_counts()
like_percent = pd.DataFrame((ca['likes'] / ca['views']) * 100)

dislike_percent = pd.DataFrame((ca['dislikes'] / ca['views']) * 100)

ca['like_percent'] = like_percent

ca['dislike_percent'] = dislike_percent
ca['positive_or_negative'] = ca['like_percent'] > ca['dislike_percent']
# comparing the likes to dislikes 

video_feel = []

for item in ca['positive_or_negative']:

    if item == True:

        video_feel.append(1)

    else:

        video_feel.append(0)
ca['positive_or_negative'] = video_feel
ca.head()
# reorganize df

ca = ca[['video_id','trending_date','trending_month','title','channel_title',

        'category_id','category_name','publish_time','publish_month','views','positive_or_negative',

        'likes','like_percent','dislikes','dislike_percent','comment_count','thumbnail_link','comments_disabled',

        'ratings_disabled','video_error_or_removed','description']]
ca.head()
f, ax = plt.subplots(3,1,figsize=(20,15))

g = sns.barplot(x=ca['publish_month'],y=ca['views'],data=ca,ax=ax[0])

g.set_title('Views across published months')

g1 = sns.barplot(x=ca['publish_month'],y=ca['likes'],data=ca,ax=ax[1])

g1.set_title('Likes across published months')

g2 = sns.barplot(x=ca['publish_month'],y=ca['dislikes'],data=ca,ax=ax[2])

g2.set_title('Dislikes across published months')

plt.tight_layout()
f, ax = plt.subplots(3,1,figsize=(20,15))

g = sns.barplot(x=ca['trending_month'],y=ca['views'],data=ca,ax=ax[0])

g.set_title('Views across trending months')

g1 = sns.barplot(x=ca['trending_month'],y=ca['likes'],data=ca,ax=ax[1])

g1.set_title('Likes across trending months')

g2 = sns.barplot(x=ca['trending_month'],y=ca['dislikes'],data=ca,ax=ax[2])

g2.set_title('Dislikes across trending months')

plt.tight_layout()
plt.figure(figsize=(15,10))

g = sns.barplot(x=ca['category_name'],y=ca['comment_count'],data=ca)

g.set_xticklabels(g.get_xticklabels(),rotation=30)

plt.title('Comments within Categories')

plt.tight_layout()

# it looks like non profits seem to have the most 'comment' activities
plt.figure(figsize=(15,10))

ax = sns.heatmap(ca.corr(),annot=True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
corr_list = ca[['views','likes','dislikes','comment_count']]
plt.figure(figsize=(15,10))

ax = sns.heatmap(data=corr_list.corr(),cmap='YlGnBu',annot=True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

f, ax = plt.subplots(2,2,figsize=(20,15))

# 1st chart (likes : views)

g = sns.scatterplot(x=ca['likes'],y=ca['views'],data=ca,ax=ax[0][0])

g.set_title('Correlation of Likes and Views')

# 2nd chart (likes : communt_count)

g1 = sns.scatterplot(x=ca['likes'],y=ca['comment_count'],data=ca,ax=ax[0][1])

g1.set_title('Correlation of Likes and Comment Counts')

# 3rd chart (comment_count : dislikes)

g2 = sns.scatterplot(x=ca['dislikes'],y=ca['comment_count'],data=ca,ax=ax[1][0])

g2.set_title('Correlation of Dislikes and Comment Counts')

# 4th chart (likes : dislikes)

g3 = sns.scatterplot(x=ca['likes'],y=ca['dislikes'],data=ca,ax=ax[1][1])

g3.set_title('Correlation of Likes and Dislikes')
plt.figure(figsize=(15,10))

g = sns.boxplot(x=ca['category_name'],y=np.log(ca['views']),data=ca)

g.set_xticklabels(g.get_xticklabels(),rotation=30)

plt.title('Views across categories')

plt.show()
f, ax = plt.subplots(2,1,figsize=(15,20))

g1 = sns.boxplot(x=ca['category_name'],y=(np.log(ca['like_percent'])),data=ca,ax=ax[0])

g1.set_xticklabels(g.get_xticklabels(),rotation=30)

g1.set_title('Like % across categories')

plt.tight_layout()

g2 = sns.boxplot(x=ca['category_name'],y=(np.log(ca['dislike_percent'])),data=ca,ax=ax[1])

g2.set_xticklabels(g.get_xticklabels(),rotation=30)

plt.title('Dislike % amongst categories')

plt.tight_layout()
plt.figure(figsize=(15,10))

g = sns.barplot(x=ca['category_name'],y=ca['views'],hue=ca['positive_or_negative'],data=ca)

g.set_xticklabels(g.get_xticklabels(),rotation=30)

plt.title('Positive vs. Negative Videos amongst Categories')

plt.tight_layout()
ca.head()
# so this gives us the hour, min, and seconds of the videos that are published

ca[['hours','minutes','seconds']] = ca['publish_time'].dt.time.astype(str).str.split(':', expand=True)
f, ax = plt.subplots(2,2,figsize=(25,20))

g = sns.lineplot(x=ca['hours'],y=ca['views'],data=ca,marker = 'o',ci=None,ax=ax[0][0])

g.set_title('Views across hours')

g1 = sns.lineplot(x=ca['hours'],y=ca['likes'],data=ca,marker = 'o',ci=None,ax=ax[0][1])

g1.set_title('Likes across hours')

g2 = sns.lineplot(x=ca['hours'],y=ca['dislikes'],data=ca,marker = 'o',ci=None,ax=ax[1][0])

g2.set_title('Dislikes across hours')

g3 = sns.lineplot(x=ca['hours'],y=ca['comment_count'],data=ca,marker = 'o',ci=None,ax=ax[1][1])

g3.set_title('Comment counts across hours')
f, ax = plt.subplots(2,1,figsize=(20,15))

g = sns.boxplot(x=ca['publish_month'],y=np.log(ca['likes']),data=ca,ax=ax[0])

g.set_title('Published videos likes by Month')

g1 = sns.boxplot(x=ca['publish_month'],y=np.log(ca['views']),data=ca,ax=ax[1])

g1.set_title('Amount of views by Month')
f, ax = plt.subplots(2,1,figsize=(20,15))

g = sns.boxplot(x=(ca['hours']),y=np.log(ca['likes']),data=ca,ax=ax[0])

g.set_title('Distribution of likes by Hour')

g1 = sns.boxplot(x=ca['hours'],y=np.log(ca['views']),data=ca,ax=ax[1])

g1.set_title('Distribution of views by Hour')
ca.head(1)
ca.describe()
ca.columns

# our x & y

# the y will be -> likes

# the x will be -> trending month, publish month, views, disklikes, and comment_count
df = ca[['category_id','trending_date','trending_month','publish_time','publish_month','category_id','views','likes','dislikes',

         'positive_or_negative','comment_count']]
df['views_log'] = np.log(df['views'])

df['likes_log'] = np.log(df['likes'])

df['dislikes_log'] = np.log(df['dislikes'])

df['comment_count_log'] = np.log(df['comment_count'])
df.head()
X = df[['trending_month','publish_month','category_id','positive_or_negative',

        'views','dislikes','comment_count']]

y = df['likes']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
fit_model = lm.fit(X_train,y_train)
print(lm.intercept_)
lm.coef_
X_train.columns
cdf = pd.DataFrame(lm.coef_,index = X_train.columns,columns=['Coefficients'])
cdf
predictions = lm.predict(X_test)
predictions
plt.figure(figsize=(15,10))

g = sns.regplot(y_test,predictions,fit_reg=True)

plt.xlabel('Y Test (Actual)')

plt.ylabel('Predictions')

g.set(xlim=(-100000,5000000),ylim=(-100000,5000000))

plt.show(g)
residuals = (y_test - predictions)

print(residuals)
residuals.mean()
g = sns.distplot(residuals,bins=30)

g.set(xlim=(-500000,500000))
plt.figure(figsize=(15,10))

sns.regplot(x=predictions,y=residuals,fit_reg=True)
from sklearn import metrics
lm.score(X_test,y_test)
print('MAE :',metrics.mean_absolute_error(y_test,predictions))

print('MSE :',metrics.mean_squared_error(y_test,predictions))

print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,predictions)))