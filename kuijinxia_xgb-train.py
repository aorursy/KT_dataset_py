import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
import time
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,log_loss
import sklearn
from sklearn.ensemble import RandomForestClassifier
import itertools
import xgboost as xgb
import random
import datetime
from wordcloud import WordCloud
import re

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

pd.options.mode.chained_assignment = None  # default='warn'
%matplotlib inline
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
target = 'interest'

merge:pd.DataFrame = pd.concat([train, test], axis=0)

X_origin = pd.DataFrame([])
X_featured = pd.DataFrame([])
X_vectorized = pd.DataFrame([])

features_to_use = np.array([])
features_to_encode = np.array([])
features_to_vectorized = np.array([])


merge.drop(['listing_id'], axis=1, inplace=True)
# merge.drop(merge.index[merge['price']>135000], axis=0, inplace=True)
merge['price'] = merge['price'].clip(lower=500, upper=135000)

merge.reset_index(drop=True,inplace=True)
nrow_train = merge.index[~merge['interest_level'].isnull()].max() + 1
merge.sample(3)
merge.loc[merge['bathrooms'] > 7 , 'bathrooms'] = 7
merge['rooms'] = merge['bathrooms'] + merge['bedrooms']
merge['rooms_diff'] = merge['bathrooms'] - merge['bedrooms']
merge['half_bathrooms'] = ((merge['rooms'] - np.floor(merge['rooms'])) > 0).astype(int)
features_to_use = np.concatenate([features_to_use, ['bathrooms', 'bedrooms', 'rooms', 'rooms_diff', 'half_bathrooms']])
features_to_use = np.unique(features_to_use)
features_to_use
merge.loc[merge['latitude'] < 1, 'latitude'] = merge['latitude'].mode()[0]
merge.loc[merge['longitude']>-1, 'longitude'] = merge['longitude'].mode()[0]
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))

merge['latitude'] = scaler.fit_transform(np.array(merge['latitude']).reshape(-1,1))
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
merge['longitude'] = scaler.fit_transform(np.array(merge['longitude']).reshape(-1,1))

merge["pos"] = merge.longitude.round(3).astype(str) + '_' + merge.latitude.round(3).astype(str)
pos_vc = merge['pos'].value_counts()
d_pos_vc = pos_vc.to_dict()
merge['density'] = merge["pos"].apply(lambda x: d_pos_vc.get(x, pos_vc.min()))

features_to_use = np.concatenate([features_to_use, ['latitude', 'longitude', 'density']])
features_to_use = np.unique(features_to_use)
features_to_use
merge['num_description_len'] = merge['description'].str.len()
merge['num_description_words'] = merge['description'].apply(lambda x:len(x.split(' ')))
merge['price_per_bedrooms'] = merge['price']/merge['bedrooms']
merge['price_per_bathrooms'] = merge['price']/merge['bathrooms']
merge['price_per_rooms'] = merge['price']/merge['rooms']
merge['beds_percent'] = merge['bedrooms']/merge['rooms']
merge['num_capital_letters'] = merge['description'].apply(lambda x: sum(1 for c in x if c.isupper()))
merge['num_address_len'] = merge['display_address'].str.len()
merge['num_address_words'] = merge['display_address'].apply(lambda x:len(x.split(' ')))
merge['address_east'] = merge['street_address'].apply(lambda x: x.find('East') > -1).astype(int)
merge['address_west'] = merge['street_address'].apply(lambda x: x.find('West') > -1).astype(int)
merge['num_photos'] = merge['photos'].str.len()
merge['num_features'] = merge['features'].str.len()
merge['num_photos_low'] = merge['num_photos'].apply(lambda x:1 if x > 22 else 0)  # all is low
merge['price_low_medium'] = merge['price'].apply(lambda x:1 if 7500< x < 10000 else 0)  # all is low or medium
merge['price_low'] = merge['price'].apply(lambda x:1 if x >= 10000 else 0)  # all is low
def cap_share(x):
    return sum(1 for c in x if c.isupper())/float(len(x) + 1)
merge['num_cap_share'] = merge['description'].apply(cap_share)
merge['num_description_lines'] = merge['description'].apply(lambda x: x.count('<br /><br />'))
merge['num_redacted'] = 0
merge['num_redacted'].ix[merge['description'].str.contains('website_redacted')] = 1
merge['num_email'] = 0
merge['num_email'].ix[merge['description'].str.contains('@')] = 1

reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
def try_and_find_nr(description):
    if reg.match(description) is None:
        return 0
    return 1
merge['num_phone_nr'] = merge['description'].apply(try_and_find_nr)



features_to_use = np.concatenate([features_to_use, ['num_description_len', 'num_description_words',
                                               'price_per_bedrooms', 'price_per_bathrooms', 'price_per_rooms', 'num_photos', 'num_features',
                                               'num_photos_low', 'price_low_medium', 'price_low',
                                                   'beds_percent', 'num_capital_letters', 'num_address_len',
                                                    'num_address_words', 'address_east', 'address_west',
                                                   'num_cap_share', 'num_description_lines', 
                                                    'num_redacted', 'num_email', 'num_phone_nr']])
features_to_use = np.unique(features_to_use)
features_to_use
def count_target_by_features(feature_name):
    global features_to_use
    merge_train = merge[:nrow_train]
    merge_test = merge[nrow_train:]
    index=list(range(merge_train.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(merge_train)
    b=[np.nan]*len(merge_train)
    c=[np.nan]*len(merge_train)

    for i in range(5):
        building_level={}
        for j in merge_train[feature_name].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*merge_train.shape[0])/5):int(((i+1)*merge_train.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=merge_train.iloc[j]
            if temp['interest_level']=='low':
                building_level[temp[feature_name]][0]+=1
            if temp['interest_level']=='medium':
                building_level[temp[feature_name]][1]+=1
            if temp['interest_level']=='high':
                building_level[temp[feature_name]][2]+=1
        for j in test_index:
            temp=merge_train.iloc[j]
            if sum(building_level[temp[feature_name]])!=0:
                bsum = sum(building_level[temp[feature_name]])
                bsum = (bsum if bsum >0 else 1)
                a[j]=building_level[temp[feature_name]][0]*1.0/bsum
                b[j]=building_level[temp[feature_name]][1]*1.0/bsum
                c[j]=building_level[temp[feature_name]][2]*1.0/bsum
    # merge[:nrow_train]['manager_level_low']=a
    # merge[:nrow_train]['manager_level_medium']=b
    # merge[:nrow_train]['manager_level_high']=c



    a1=[]
    b1=[]
    c1=[]
    building_level={}
    for j in merge_train[feature_name].values:
        building_level[j]=[0,0,0]
    for j in range(merge_train.shape[0]):
        temp=merge_train.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp[feature_name]][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp[feature_name]][1]+=1
        if temp['interest_level']=='high':
            building_level[temp[feature_name]][2]+=1

    for i in merge_test[feature_name].values:
        if i not in building_level.keys():
            a1.append(np.nan)
            b1.append(np.nan)
            c1.append(np.nan)
        else:
            bsum = sum(building_level[i])
            bsum = (bsum if bsum >0 else 1)
            a1.append(building_level[i][0]*1.0/bsum)
            b1.append(building_level[i][1]*1.0/bsum)
            c1.append(building_level[i][2]*1.0/bsum)
    merge[feature_name+ '_low']= np.concatenate([a , a1])
    merge[feature_name+ '_medium']= np.concatenate([b , b1])
    merge[feature_name+ '_high']= np.concatenate([c , c1])
    merge[feature_name+ '_low'].fillna(0, inplace=True)
    merge[feature_name+ '_medium'].fillna(0, inplace=True)
    merge[feature_name+ '_high'].fillna(0, inplace=True)

    print(feature_name)
    features_to_use = np.concatenate([features_to_use, [feature_name+ '_low',feature_name+ '_medium',
                                                   feature_name+ '_high']])
    features_to_use = np.unique(features_to_use)
count_features_name = ['building_id', 'manager_id', 'display_address', 'street_address']
for feature in count_features_name:
    count_target_by_features(feature)
features_to_use
interest_level_dict = {'low' : 0, 'medium' : 1, 'high' : 2 }
merge['interest'] = merge['interest_level'].map(interest_level_dict)


created_time = pd.to_datetime(merge['created'],format='%Y-%m-%d %H:%M:%S')
merge['month'] = created_time.dt.month
merge['day'] = created_time.dt.day
merge['hour'] = created_time.dt.hour
merge['weekday'] = created_time.dt.weekday
merge['week'] = created_time.dt.week
merge['quarter'] = created_time.dt.quarter
merge['weekend'] = ((merge['weekday'] == 5) | (merge['weekday'] == 6))
merge['days_since'] = created_time.max() - created_time
merge['days_since'] = (merge['days_since'] / np.timedelta64(1, 'D')).astype(int)

features_to_encode = np.concatenate([features_to_encode, ['month', 'day', 'hour', 'weekday', 'week', 'quarter', 'hour', 'weekend']])
features_to_encode = np.unique(features_to_encode)
features_to_encode

display_address_min_df = 10
street_address_min_df = 10
features_min_df = 10
description_max_features = 20
cv = CountVectorizer(min_df=display_address_min_df)
X_display_address = cv.fit_transform(merge['display_address'])

cv = CountVectorizer(min_df=street_address_min_df)
X_street_address = cv.fit_transform(merge['street_address'])



merge['features_'] = merge['features'].apply(lambda x:' '.join(['_'.join(k.split(' ')) for k in x]))
cv = CountVectorizer(stop_words='english', max_features=200)
X_features = cv.fit_transform(merge['features_'])


tv = TfidfVectorizer(max_features=description_max_features,
                    ngram_range=(1, 5),
                    stop_words='english')
X_description = tv.fit_transform(merge['description'])

X_vectorized = hstack((X_display_address, X_street_address, X_features, X_description)).tocsr()
ohe = sklearn.preprocessing.OneHotEncoder()
X_encode = ohe.fit_transform(merge[features_to_encode])
def union_features(features_to_use, X_encode, X_vectorized, target, nrow_train):
    X_origin = merge[features_to_use]
    X_origin.fillna(0 ,inplace=True)
    X = hstack((X_origin, X_encode, X_vectorized)).tocsr()
    y = merge[target]

    X_train_all = X[:nrow_train]
    X_test = X[nrow_train:]
    y_train_all = y[:nrow_train]
    # y_test = y[nrow_train:]


    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2 , random_state=10)
    return X_train, X_test, X_val, y_train, y_val 

X_train, X_test, X_val, y_train, y_val =  union_features(features_to_use, X_encode, X_vectorized, target, nrow_train)
merge.info()

def runXGB(X_train, y_train, X_val, y_val, model=None):
    start_time = time.time()
    if model==None:
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'),(d_valid, 'valid')]

        seed_val = 10
        params = {}
        params['objective'] = 'multi:softprob'
        params['eta'] = 0.05
        params['max_depth'] = 6
        params['silent'] = 1
        params['num_class'] = 3
        params['eval_metric'] = "mlogloss"
        params['min_child_weight'] = 1
        params['subsample'] = 0.7
        params['colsample_bytree'] = 0.7
        params['seed'] = seed_val
        model = xgb.train(params, dtrain=d_train, num_boost_round=3200, evals = watchlist,
                          early_stopping_rounds=100, maximize=False, verbose_eval=40)
  
    print('[{}] Finished fit'.format(time.time() - start_time))
    print(' model best_score:%.5f' % model.best_score)
    preds = model.predict(xgb.DMatrix(X_val))

    print('[{}] Finished predict'.format(time.time() - start_time))

    start = 0
    end = X_val.shape[0]
    rand_sample = np.random.randint(start, end, 10)
    ten_sample = pd.DataFrame(np.array([list(np.array(y_val)[rand_sample]) , list(np.argmax(preds[rand_sample], axis=1))]).T, columns=['true', 'predict'])
    print('[{}] Finished '.format(time.time() - start_time))
    print(ten_sample)
    true_preds = pd.DataFrame(np.array([np.array(y_val), np.argmax(preds, axis=1)]).T, columns=['true','pre']) 
    return true_preds, model
  
true_preds, model = runXGB(X_train, y_train, X_val, y_val)
Y_pred = model.predict(xgb.DMatrix(X_test))
ids = np.array(test['listing_id'])
preds = pd.DataFrame({"listing_id": ids, "high":Y_pred[:, 0],
                      "medium":Y_pred[:, 1], "low":Y_pred[:, 2]})
preds.to_csv('my_submission.csv' ,index=False)