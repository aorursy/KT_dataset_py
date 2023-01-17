import os

import sys

import operator

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import xgboost as xgb

import random

from sklearn import model_selection, preprocessing, ensemble

from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#input data

train_df=pd.read_json('../input/train.json')

test_df=pd.read_json('../input/test.json')

train_df.head()



#removing outliers 



test_df["bathrooms"].loc[19671] = 1.5

test_df["bathrooms"].loc[22977] = 2.0

test_df["bathrooms"].loc[63719] = 2.0



train_df["price"] = train_df["price"].clip(upper=13000)

test_df["price"] = test_df["price"].clip(upper=13000)



ulimit = np.percentile(train_df.price.values, 99)

train_df['price'].ix[train_df['price']>ulimit] = ulimit



ulimit = np.percentile(test_df.price.values, 99)

test_df['price'].ix[test_df['price']>ulimit] = ulimit



train_df["logprice"] = np.log(train_df["price"])

test_df["logprice"] = np.log(test_df["price"])

# count of "photos"

train_df["num_photos"] = train_df["photos"].apply(len)

test_df["num_photos"] = test_df["photos"].apply(len)



train_df["num_features"] = train_df["features"].apply(len)

test_df["num_features"] = test_df["features"].apply(len)



train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))

test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))



train_df["pos"] = train_df.longitude.round(3).astype(str) + '_' + train_df.latitude.round(3).astype(str)

test_df["pos"] = test_df.longitude.round(3).astype(str) + '_' + test_df.latitude.round(3).astype(str)



vals = train_df['pos'].value_counts()

dvals = vals.to_dict()

train_df["density"] = train_df['pos'].apply(lambda x: dvals.get(x, vals.min()))

test_df["density"] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))



#basic features

train_df["price_t"] =train_df["price"]/train_df["bedrooms"]

test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 

train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 

test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 



train_df["photo_per_room"]=train_df["num_photos"]/train_df["room_sum"]

test_df["photo_per_room"]=test_df["num_photos"]/test_df["room_sum"]



train_df["photo_per_price"]=train_df["num_photos"]/train_df["price"]

test_df["photo_per_price"]=test_df["num_photos"]/test_df["price"]



train_df['price_per_room'] = train_df['price']/train_df['room_sum']

test_df['price_per_room'] = test_df['price']/test_df['room_sum']



# can we contact someone via e-mail to ask for the details?

train_df['num_email'] = 0

train_df['num_email'].ix[train_df['description'].str.contains('@')] = 1

test_df['num_email'] = 0

test_df['num_email'].ix[test_df['description'].str.contains('@')] = 1

    

#and... can we call them?

    

reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)

def try_and_find_nr(description):

    if reg.match(description) is None:

        return 0

    return 1



train_df['num_phone_nr'] = train_df['description'].apply(try_and_find_nr)

test_df['num_phone_nr'] = test_df['description'].apply(try_and_find_nr)



features_to_use=["bathrooms", "bedrooms", "price_t","room_sum","latitude","longitude","num_photos","density","logprice","num_features","num_description_words","price_per_room","listing_id","photo_per_room","photo_per_price","num_email","num_phone_nr"]

print(train_df['price'].head())





def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.03

    param['max_depth'] = 3

    param['silent'] = 0

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    param['n_estimators'] = 500

    num_rounds = num_rounds



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model

	
index=list(range(train_df.shape[0]))

	

random.shuffle(index)

a=[np.nan]*len(train_df)

b=[np.nan]*len(train_df)

c=[np.nan]*len(train_df)



for i in range(5):

    building_level={}

	#initializing

    for j in train_df['manager_id'].values:

        building_level[j]=[0,0,0]

	#Splitting into 2 groups

    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]

    train_index=list(set(index).difference(test_index))

    for j in train_index:

        temp=train_df.iloc[j]

        if temp['interest_level']=='low':

            building_level[temp['manager_id']][0]+=1

        if temp['interest_level']=='medium':

            building_level[temp['manager_id']][1]+=1

        if temp['interest_level']=='high':

            building_level[temp['manager_id']][2]+=1

    for j in test_index:

        temp=train_df.iloc[j]

        if sum(building_level[temp['manager_id']])!=0:

            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])

            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])

            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])

train_df['manager_level_low']=a

train_df['manager_level_medium']=b

train_df['manager_level_high']=c



a=[]

b=[]

c=[]

building_level={}

for j in train_df['manager_id'].values:

    building_level[j]=[0,0,0]

for j in range(train_df.shape[0]):

    temp=train_df.iloc[j]

    if temp['interest_level']=='low':

        building_level[temp['manager_id']][0]+=1

    if temp['interest_level']=='medium':

        building_level[temp['manager_id']][1]+=1

    if temp['interest_level']=='high':

        building_level[temp['manager_id']][2]+=1



for i in test_df['manager_id'].values:

    if i not in building_level.keys():

        a.append(np.nan)

        b.append(np.nan)

        c.append(np.nan)

    else:

        a.append(building_level[i][0]*1.0/sum(building_level[i]))

        b.append(building_level[i][1]*1.0/sum(building_level[i]))

        c.append(building_level[i][2]*1.0/sum(building_level[i]))

test_df['manager_level_low']=a

test_df['manager_level_medium']=b

test_df['manager_level_high']=c



features_to_use.append('manager_level_low') 

features_to_use.append('manager_level_medium') 

features_to_use.append('manager_level_high')

categorical = ["display_address", "manager_id", "building_id", "street_address"]



for f in categorical:

        if train_df[f].dtype=='object':

            #print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train_df[f].values) + list(test_df[f].values))

            train_df[f] = lbl.transform(list(train_df[f].values))

            test_df[f] = lbl.transform(list(test_df[f].values))

            features_to_use.append(f)

print(train_df["features"].head())

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))



print(train_df["manager_level_low"].head())
from scipy import sparse

tfidf = CountVectorizer(min_df=5)

tr_sparse = tfidf.fit_transform(train_df["features"])

te_sparse = tfidf.transform(test_df["features"])

print(te_sparse)
print(test_df[features_to_use].head())

print(train_df[features_to_use].head())

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()

test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()



target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, test_X.shape)
cv_scores = []

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)

for dev_index, val_index in kf.split(range(train_X.shape[0])):

        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]

        dev_y, val_y = train_y[dev_index], train_y[val_index]

        preds, model = runXGB(dev_X, dev_y, val_X, val_y)

        cv_scores.append(log_loss(val_y, preds))

        print(cv_scores)

        break
preds, model = runXGB(train_X, train_y, test_X, num_rounds=1500)

out_df = pd.DataFrame(preds)

out_df.columns = ["high", "medium", "low"]

out_df["listing_id"] = test_df.listing_id.values

out_df.to_csv("xgb_starter2.csv", index=False)