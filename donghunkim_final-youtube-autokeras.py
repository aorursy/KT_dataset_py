import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as snsn

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
idx=(data['video_error_or_removed']==False) & (data['ratings_disabled']==False) & (data['comments_disabled']==False)

data=data.loc[idx,:]
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
data_dr=data_dr.drop(columns=['publish_time','id','thumbnail_link','description'])
data_dr.info()
data_dr['category_id']=data_dr['category_id'].astype('object')
data_dr.info()
from sklearn.model_selection import train_test_split
data_num_cols = ['likes','comment_count','dislikes','desc_length','trending_count','title_length','tag_count','to_trending_days']

data_cols = ['likes','comment_count','dislikes','desc_length','trending_count','title_length','tag_count','to_trending_days','category_id']





Xn_train, Xn_test, yn_train, yn_test = train_test_split(data_dr[data_num_cols], data_dr['views'], test_size=0.2, random_state=1)



X_train, X_test, y_train, y_test = train_test_split(data_dr[data_cols], data_dr['views'], test_size=0.2, random_state=1)
X_train.head()
!pip install autokeras 
#category_id 추가

data_num_type = (len(data_num_cols)) * ['numerical']

data_num_type = dict(zip(data_num_cols, data_num_type))



data_type = (len(data_cols)-1) * ['numerical'] + ['categorical']

data_type = dict(zip(data_cols, data_type))
import autokeras as ak

import tensorflow as tf

import sklearn.metrics as metrics
%%time

regressor = ak.StructuredDataRegressor(max_trials=10, column_names=data_num_cols, column_types=data_num_type)

regressor.fit(x=Xn_train, y=yn_train, epochs=50)

auto_y_predicted = regressor.predict(Xn_test)
print('AUTO ML:\n',

      'MAE =',round(metrics.mean_absolute_error(y_test, auto_y_predicted),0), 

      ',MSE = ',round(metrics.mean_squared_error(y_test, auto_y_predicted),0), 

      ',RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, auto_y_predicted)),0)  )
model = regressor.export_model()

tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
%%time

regressor = ak.StructuredDataRegressor(max_trials=10, column_names=data_cols, column_types=data_type)

regressor.fit(x=X_train, y=y_train, epochs=50)

auto_y_predicted = regressor.predict(X_test)
print('AUTO ML:\n',

      'MAE =',round(metrics.mean_absolute_error(y_test, auto_y_predicted),0), 

      ',MSE = ',round(metrics.mean_squared_error(y_test, auto_y_predicted),0), 

      ',RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, auto_y_predicted)),0)  )
model = regressor.export_model()

tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)