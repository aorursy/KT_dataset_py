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
data=pd.read_csv("../input/youtube-new/KRvideos.csv", engine="python")
data=data.copy()
data.info()
data.head()
cat_data=pd.read_json("../input/youtube-new/KR_category_id.json")
cat_items=cat_data['items']
cat_items.count()
for idx in range(0, cat_items.count()):
    cat_data.loc[idx,'id'] = cat_items[idx]['id']
    cat_data.loc[idx,'category'] = cat_items[idx]['snippet']['title']
cat_data=cat_data.drop(columns=['kind','etag','items'])
cat_data.info()
cat_data.head()
cat_data['id']=cat_data['id'].astype('int64')
data=pd.merge(data, cat_data, left_on='category_id', right_on='id', how='left')
data.info()
data['category_id'].loc[data['id'].isnull()==True].value_counts()
data['id'].fillna(29, inplace=True)
data['category'].fillna('Nonprofits & Activism', inplace=True)
data.info()
data['video_id'].describe()
idx=(data['video_id']!='#NAME?')
data=data.loc[idx,:]
data['video_id'].describe()
data['trending_date'].head()
data['trending_date']=pd.to_datetime(data['trending_date'], format='%y.%d.%m').dt.date
data['publish_time'].head()
data[['publish_date','publish_time']]=data['publish_time'].str.split('T', expand=True)
data[['publish_date','publish_time']].head()
data['publish_date']=pd.to_datetime(data['publish_date']).dt.date
data.info()
data.sort_values(by='trending_date', inplace=True)
data_dr=data.drop_duplicates('video_id', keep='first')
data_dr['video_id'].describe(include='all')
data_dr.info()
data_dr=data_dr.reset_index(drop=True)
data_dr['tags'].head()
data_dr['tags']=data_dr['tags'].str.replace(pat=r'|', repl=r' ', regex=True)
data_dr['tags']=data_dr['tags'].str.replace(pat=r'[^\w\s]', repl=r'', regex=True)
data_dr['tags'].head()
data_dr['tag_list']=data_dr['tags'].str.split(" ")
data_dr['tag_list'].head()
data_dr['tag_list']=data_dr['tag_list'].apply(lambda x: list(set(x)))
data_dr['tag_list'].head()
tag_sum=[]
for i in range(0, data_dr['tag_list'].count()):
    tag_sum = tag_sum + data_dr.loc[i, 'tag_list']
count={}
for i in tag_sum:
    try: count[i] += 1
    except: count[i]=1
df_count = pd.DataFrame(count, index=['tag_count'])
df_count = df_count.T
#df_count=df_count.reset_index().rename(columns={"index": "tag"})
df_count.head()
df_count=df_count.drop('none') # none 데이터 제거
df_count.sort_values(by=['tag_count'], ascending=False, axis=0)
tag_idx=(df_count['tag_count']>10)
df_count=df_count.loc[tag_idx,:]
c_index=list(df_count.index)
df_count.describe()
df_count['tag_views']=0
df_count['tag_like']=0
df_count['tag_dislike']=0
df_count['tag_comment']=0

for tag in c_index:
    df_count.loc[tag, 'tag_views']=data_dr.loc[data_dr['tags'].str.contains(tag),'views'].mean()
    df_count.loc[tag, 'tag_like']=data_dr.loc[data_dr['tags'].str.contains(tag),'likes'].mean()
    df_count.loc[tag, 'tag_dislike']=data_dr.loc[data_dr['tags'].str.contains(tag),'dislikes'].mean()
    df_count.loc[tag, 'tag_comment']=data_dr.loc[data_dr['tags'].str.contains(tag),'comment_count'].mean()


tag

    
#df_count.loc[tag, 'tag_views']=data_dr.loc[data_dr['tags'].str.contains(tag),'views'].mean()

#df_count.loc[tag,:]

#data_dr.loc[data_dr['tags'].str.contains(tag),'views']

#for tag in c_index:
#    df_count.loc[tag,'tag_views']=data_dr[data_dr['tag_list'].contains(tag),'views'].mean()
#data_dr.loc[data_dr['tags'].str.contains('윤종신'),'views'].mean()
df_count.describe()
df_count.head()
df_count.sort_values(by=['tag_count'], ascending=False, axis=0)
df_count.sort_values(by=['tag_views'], ascending=False, axis=0)
df_count.sort_values(by=['tag_like'], ascending=False, axis=0)
df_count.sort_values(by=['tag_dislike'], ascending=False, axis=0)
df_count.sort_values(by=['tag_comment'], ascending=False, axis=0)