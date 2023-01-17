!pip install xgboost
import numpy as np

import pandas as pd
data=pd.read_csv("../input/youtube-new/KRvideos.csv", engine="python")

#data = org_data.copy()
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
idx=(data['video_error_or_removed']==False) & (data['ratings_disabled']==False) # & (data['comments_disabled']==False)

data=data.loc[idx,:]
data[['comments_disabled','ratings_disabled','video_error_or_removed']].describe()
data=data.drop(columns=['comments_disabled','ratings_disabled','video_error_or_removed'])
data['video_id'].describe()
idx=(data['video_id']!='#NAME?')

data=data.loc[idx,:]
data['video_id'].describe()
#category_id: str 타입변환

data['category_id'] = data['category_id'].astype(str)
# category_id, 

# tags, title, chanel_title => text analysis 

# publish_time, trending_date  => 날짜 차이필요

#

# likes, dislike, comment_count : X,Y의 선후 인과 관계 문제

# thumbnail_link : 불필요 컬럼



# X = 

# Y = view

data.info()
#tag 데이터 list변환

data['tag_list'] = data['tags'].apply(lambda s : s.replace('"','').split('|') )

data['tag_corpus'] = data['tags'].apply(lambda s : s.replace('"','').replace(' ','').replace('|',' ')) #따움표,공백제거 후 corpus 생성
tag_data_count = {}



for tags in data['tag_list']:

    for tag in tags:

        if tag in tag_data_count:

            tag_data_count[tag] += 1

        else:

            tag_data_count[tag] = 1
tag_data_items = sorted(tag_data_count.items(), key= lambda x : x[1], reverse=True) 

df_tag = pd.DataFrame(tag_data_items,columns=['tag','tag_cnt'])

df_tag.head() 
df_tag.shape
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc

import os

%matplotlib inline



#plt.rcParams['axes.unicode_minus'] = False

#fontpath = '../input/hangul-font/gulim.ttc'

#fontprop = font_manager.FontProperties(fname=fontpath, size=12)

#plt.rc('font', family=fontprop.get_name())
#sns.barplot(x=df_tag['tag'][:100],y=df_tag['tag_cnt'][:100])

#plt.show()
data.reset_index(inplace=True)
cols = ['likes','dislikes','comment_count']

X = data[cols].copy()

y = data['views']
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #80% , 20%
#sns.pairplot(data, x_vars=['likes','dislikes','comment_count'], y_vars='views', height=5, aspect=0.7, kind='reg')

#plt.show()
line_fitter = LinearRegression()

line_fitter.fit(X_train, y_train)

y_predicted = line_fitter.predict(X_test)
print('R^2=', line_fitter.score(X_train,y_train))

print('coef=',line_fitter.coef_, ', intercept=', line_fitter.intercept_)
# calculate MAE, MSE, RMSE

print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))

print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))

print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))
#none 부분 제거, 상위 500

df_tag_freq = df_tag[1:2000]
#sns.barplot(x=df_tag_freq['tag'][:],y=df_tag_freq['tag_cnt'][:])

#plt.show()
from sklearn.feature_extraction.text import CountVectorizer

#corpus = data['tag_corpus'].values

corpus = df_tag_freq['tag'].values

vect = CountVectorizer()

vect.fit(corpus)

vect.vocabulary_
data['tag_corpus']
tag_vect = vect.transform(data['tag_corpus'])

tag_vect #32704 x 62870
data.shape
tag_array = tag_vect.toarray()
df_tag_vect = pd.DataFrame(tag_array)
df_tag_vect.sum(axis=1)
XX = pd.concat([X,df_tag_vect],axis=1)
line_fitter = LinearRegression()

line_fitter.fit(X_train, y_train)

y_predicted = line_fitter.predict(X_test)
sns.scatterplot(y_test, y_predicted)

plt.show()
# calculate MAE, MSE, RMSE

print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))

print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))

print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))
import xgboost
model = xgboost.XGBRegressor(learning_rate=0.1,max_depth=5, n_estimators=100,random_state=0) 

model.fit(X_train,y_train)
y_predicted = model.predict(X_test)

sns.scatterplot(y_test, y_predicted)

plt.show()
# calculate MAE, MSE, RMSE

print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))

print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))

print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100,max_depth=5, random_state=0)

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

sns.scatterplot(y_test, y_predicted)

plt.show()
# calculate MAE, MSE, RMSE

print('MAE =',round(metrics.mean_absolute_error(y_test, y_predicted),3))

print('MSE = ',round(metrics.mean_squared_error(y_test, y_predicted),3))

print('RMSE = ',round(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)),3))