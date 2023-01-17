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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

file_name='../input/youtube-new/USvideos.csv'

my_df=pd.read_csv(file_name,index_col='video_id')

my_df.head(3)
my_df['tresnding_date']=pd.to_datetime(my_df['trending_date'],format='%y.%d.%m')

my_df['trending_date'].head()
my_df['publish_time']=pd.to_datetime(my_df['publish_time'],format='%Y-%m-%dT%H:%M:%S.%fZ')

my_df['publish_time'].head()
my_df['publish_time'].dt.date
my_df.insert(4,'publish_date',my_df['publish_time'].dt.date)

my_df['publish_time']=my_df['publish_time'].dt.time
my_df[['publish_time','publish_date']].head()
type_int=['views','likes','dislikes','comment_count']

for i in type_int:

    my_df[i]=my_df[i].astype(int)



my_df['category_id']=my_df['category_id'].astype(str)
import json

id_to_category = {}

with open('../input/youtube-new/US_category_id.json','r') as f:

    data=json.load(f)

    for category in data['items']:

        id_to_category[category['id']]=category['snippet']['title']

       # print(category['id'])

id_to_category
my_df.insert(4,'category',my_df['category_id'].map(id_to_category))

my_df[['category_id','category']].head()
corr_matrix=my_df[['views', 'likes', 'dislikes', 'comment_count']].corr()

corr_matrix
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10, 10)

from matplotlib import cm

fig,ax=plt.subplots()

heatmap=ax.imshow(corr_matrix,interpolation='nearest',cmap=cm.coolwarm)



cbar_min=corr_matrix.min().min()

cbar_max=corr_matrix.max().min()

cbar=fig.colorbar(heatmap,ticks=[cbar_min,cbar_max])

keep_columns=['views', 'likes', 'dislikes', 'comment_count']

labels = ['']

for column in keep_columns:

    labels.append(column)

    labels.append('')

ax.set_xticklabels(labels, minor=False)

ax.set_yticklabels(labels, minor=False)





plt.show()
my_df = my_df[~my_df.index.duplicated(keep='last')]
my_df.shape
my_df.head(3)
#Top 10 videos -Function

def top_10(data,column,n=10):

    sorted_data=data.sort_values(column,ascending=False).iloc[:n]

    #sns.barplot(sorted_data[column])

    ax=sorted_data[column].plot.bar()

    labels=[]

    for item in sorted_data['title']:

        labels.append(item[:8]+'..')

    ax.set_xticklabels(labels,rotation=60,fontsize=10)

    

top_10(my_df, 'views')
#Likes Vs. Views

import seaborn as sns

sns.scatterplot(x='likes',y='views',data=my_df)
top_10(my_df, 'likes', n=5) # only visualizes the top 5
top_10(my_df, 'dislikes', n=5)
top_10(my_df, 'comment_count', n=5)
category_count=my_df['category'].value_counts()

category_count
ax=category_count.plot.bar()
#Highly used tag

tag=[]

for i in my_df['tags']:

    for j in i.split('|'):

        tag.append(j.replace('"'," ").strip().replace('"',""))

        
TAGS=pd.DataFrame(tag,columns=['TAGS'])
TAGS=TAGS['TAGS'].value_counts().iloc[:6]

TAGS=TAGS.drop('[none]')

TAGS
values=[]

for i in TAGS:

    values.append(i)
labels=[]

for i in TAGS.keys():

    if i not in [False,0,"[none]"]:

        labels.append(i)

sns.barplot(x=labels,y=values)

# These are the highly used tags in the videos