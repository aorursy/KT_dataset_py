# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
us_videos=pd.read_csv('../input/USvideos.csv')
us_category=pd.read_json('../input/US_category_id.json')
us_videos.head()
us_category.head()
us_videos.describe()
cdata={}
for index, row in us_category.iterrows():
    cid=row['items'].get('id')
    title=row['items'].get('snippet').get('title')
    cdata.update({cid:title})
def compute_title(i):
    if str(i) in cdata:
        return cdata[str(i)]
    else:
        return np.n
us_videos['category_title']=us_videos['category_id'].apply(compute_title)
us_videos.head(5)
us_videos.info()

us_videos[us_videos['description'].isnull()==True].count()
plt.figure(figsize=(14,8))
sns.set_style('whitegrid')
sns.countplot(y='category_title',data=us_videos)
plt.ylabel('Category Title')
plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="likes",data=us_videos,alpha=0.5)
us_videos.drop('description',axis=1,inplace=True)
plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="comment_count",data=us_videos,alpha=0.5)
corr_df=us_videos.corr()
corr_df
plt.figure(figsize=(14,6))
sns.heatmap(corr_df,annot=True)
plt.figure(figsize=(14,6))
sns.jointplot(x='likes',y='comment_count',data=us_videos,color="green",kind='reg')
plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='likes',data=us_videos)
plt.xticks(rotation=30)
plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='dislikes',data=us_videos)
plt.xticks(rotation=30)
plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='comment_count',data=us_videos)
plt.xticks(rotation=30)