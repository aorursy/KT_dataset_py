import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import json
from sklearn.linear_model import LinearRegression
from scipy import sparse
import scipy
PATH_TO_DATA = '../input/'
data=pd.read_json(PATH_TO_DATA+'train.json', lines=True)
target=pd.read_csv(PATH_TO_DATA+'train_log1p_recommends.csv')
data['target']=target['log_recommends']
data.head(2)
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def preprocessing(df, columns=['count_by_author','images_count','content_length_type','min_reads','target','image_url',\
                               'h1_count','h2_count','h3_count','href_count','count_by_domain','domain']):
    df.image_url=np.where(df.image_url.isna(), 0, 1)
    df=df.drop('_spider', axis=1)
    df.author=df.author.apply(lambda x: x.get('url').split('@')[1])
    df['count_by_author']=df.groupby('author').transform('count')['content']  #count number of posts per author
    df['count_by_domain']=df.groupby('domain').transform('count')['content'] #count number of posts per domain
    df['images_count']=df.content.apply(lambda x: x.count('<img'))  #count number of images
    df['h1_count']=df.content.apply(lambda x: x.count('<h1')) #count number of titles
    df['h2_count']=df.content.apply(lambda x: x.count('<h2')) ##count number of titles
    df['h3_count']=df.content.apply(lambda x: x.count('<h3')) #count number of titles
    df['href_count']=df.content.apply(lambda x: x.count('href')) #count number of links
    df['content_length']=df['content'].apply(lambda x: len(strip_tags(x).split()))  #count content lenght and split to categories below
    df['content_length_type']='medium'
    df['content_length_type'][df.content_length<800]='short'
    df['content_length_type'][(df.content_length>2500) & (df.content_length<5000)]='long_read'
    df['content_length_type'][df.content_length>=5000]='huge'
    df.domain=df.domain.apply(lambda x: ' '.join(x.split('.')))
    df['min_reads']=df.meta_tags.apply(lambda x: int(x.get('twitter:data1').split()[0]))  #get read mins 
    df=df[columns]
    df=pd.get_dummies(df, columns=['content_length_type','domain'])
    #columns=['count','images_count','content_length','content_length_type','min_reads','text','target','image_url']
    return df
data_sample=data.sample(5000)
%%time
df=preprocessing(data)
df.head(2)
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(15,10) )

df.h1_count.plot.hist(ax=axes[0,0], title="h1")
df.h2_count.plot.hist(ax=axes[0,1], title="h2")
df.h3_count.plot.hist(ax=axes[0,2], title="h3")
df.images_count.plot.hist(ax=axes[1,0], title="images per post")
df.href_count.plot.hist(ax=axes[1,1], title="links per post")
df.min_reads.plot.hist(ax=axes[1,2], title="min_read")
from sklearn.preprocessing import PowerTransformer
df2=df.copy()
cols=['h1_count','h2_count','h3_count','images_count','href_count','count_by_domain','count_by_author']
df2[cols] = PowerTransformer().fit_transform(df[cols].values)

fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(15,10) )

df2.h1_count.plot.hist(ax=axes[0,0], title="h1")
df2.h2_count.plot.hist(ax=axes[0,1], title="h2")
df2.h3_count.plot.hist(ax=axes[0,2], title="h3")
df2.images_count.plot.hist(ax=axes[1,0], title="images per post")
df2.href_count.plot.hist(ax=axes[1,1], title="links per post")
df2.min_reads.plot.hist(ax=axes[1,2], title="mins read")
print(df['target'].mean())
print(df['target'].std())
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost

lgbm = LGBMRegressor(max_depth=-1, learning_rate=0.01, n_estimators=500)
rf=RandomForestRegressor(max_features=7, n_estimators=100, max_depth=7)
ridge = Ridge(random_state=17,)
kf=KFold(n_splits=5, random_state=None, shuffle=False)
scores = cross_val_score(lgbm, df.drop(['target'],axis=1), df['target'], scoring="neg_mean_absolute_error", cv=kf)
print(scores, scores.mean())
scores = cross_val_score(ridge, df.drop(['target'],axis=1), df['target'], scoring="neg_mean_absolute_error", cv=kf)
print(scores, scores.mean())
#And ridge again with normalized data

scores = cross_val_score(ridge, df2.drop(['target','h1_count'],axis=1), df['target'], scoring="neg_mean_absolute_error", cv=kf)
print(scores, scores.mean())
lgbm.fit(df.drop(['target'],axis=1), df['target'])
cols=df.drop(['target'],axis=1).columns
plt.figure(figsize=(15,5))
plt.xticks(rotation=70)
plt.bar(cols[np.argsort(lgbm.feature_importances_)[-15:]][::-1], lgbm.feature_importances_[np.argsort(lgbm.feature_importances_)[-15:]][::-1])


Well, all 