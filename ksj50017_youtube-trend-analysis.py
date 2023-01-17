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
#idx=(data['video_error_or_removed']==False) & (data['ratings_disabled']==False) & (data['comments_disabled']==False)
#data=data.loc[idx,:]
#data[['comments_disabled','ratings_disabled','video_error_or_removed']].describe()
data=data.drop(columns=['comments_disabled','ratings_disabled','video_error_or_removed'])
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
data['to_trending_days']=(data['trending_date']-data['publish_date']).dt.days
data['to_trending_days'].head()
data.info()
data['tags'].head()
data['tag_count']=data['tags'].apply(lambda x: len(x.split("|")) if x != '[none]' else 0)
data['tag_count'].head()
data['tags'].head()
data['tag_list']=data['tags'].str.split("|")
data['tag_list'].head()
df=data[['category_id','category']]
df_dummy= pd.get_dummies(df)
df_dummy.head()
video_count=data.groupby("video_id").size()
video_count=video_count.reset_index()
video_count.head()
data=pd.merge(data, video_count, on='video_id', how='left')
data.rename(columns={0:'trending_count'}, inplace=True)
data.info()
data['trending_count'].describe()
data['title_length']=data['title'].apply(lambda x: len(str(x)) if pd.isnull(x) == False else 0)
data['title_length'].describe()
data.info()
data['desc_length']=data['description'].apply(lambda x: len(str(x)) if pd.isnull(x) == False else 0)
data['desc_length'].describe()
data.info()
data.sort_values(by='trending_date', inplace=True)
data_dr=data.drop_duplicates('video_id', keep='first')
data_dr['video_id'].describe(include='all')
data_dr.info()
data_dr=data_dr.drop(columns=['publish_time','id','thumbnail_link','description'])
data_dr.info()
data_dr['category_id']=data_dr['category_id'].astype('object')
data_dr.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set()
data_dr_corr=data_dr.corr()
plt.figure(figsize = (10,7))
sns.heatmap(data_dr_corr,
            cmap='coolwarm', cbar=True, annot=True, square=True, fmt='.2f')
data_dr_corr['views'].sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
X_train, X_test, Y_train, Y_test = train_test_split(data_dr[['likes','comment_count','dislikes','desc_length','trending_count','title_length','tag_count','to_trending_days']], data_dr['views'], test_size=0.2, random_state=1)
X_train.head()
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
tree = DecisionTreeRegressor()
lr.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
lasso.fit(X_train, Y_train)
tree.fit(X_train, Y_train)
score = make_scorer(r2_score)
print('lr_r2_score : {:}'.format(score(lr, X_test, Y_test)))
print('ridge_r2_score : {:}'.format(score(ridge, X_test, Y_test)))
print('lasso_r2_score : {:}'.format(score(lasso, X_test, Y_test)))
print('tree_r2_score : {:}'.format(score(tree, X_test, Y_test)))