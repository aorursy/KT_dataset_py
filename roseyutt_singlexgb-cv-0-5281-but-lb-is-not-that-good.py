import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import string

import datetime

from sklearn.model_selection import train_test_split

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")
def load_data():

    train_path="./train.json"

    test_path="./test.json"

    train=pd.read_json(train_path)

    y=train['interest_level'].reset_index(drop=True)

    y_map = {'low': 0, 'medium': 1, 'high': 2}

    y = y.apply(lambda x: y_map[x])

    test=pd.read_json(test_path).reset_index(drop=True)

    listing_id=test.listing_id

    return train,test,y,listing_id
def process_buildingid(data,data1):

    fea=['building_id','price','manager_id','bathrooms','bedrooms','latitude','longitude']

    subdata=data[fea].reset_index(drop=True)

    subdata=pd.concat((subdata,data1[['Month','Wday','Diff_days']]),axis=1)

    t=subdata[['building_id','price']]

    t['building_count']=1

    t=t.groupby(['building_id']).agg('count').reset_index()

    t.drop('price',axis=1,inplace=True)



    t1=subdata[['building_id','price','bathrooms','bedrooms','latitude','longitude','Month','Wday','Diff_days']]

    t1['room_size']=t1['bathrooms']+t1['bedrooms']

    t1.drop(['bathrooms','bedrooms'],axis=1,inplace=True)

    

    subdata=pd.merge(subdata,t,on='building_id',how='left')

    

    status=['sum','mean','min','max','median']

    for i,statu in enumerate(status):

        temp=t1.groupby('building_id').agg(statu).reset_index()

        temp.rename(columns={'price':'build_price_{}'.format(statu),'latitude':'build_lat_{}'.format(statu),'longitude':'build_lon_{}'.format(statu),

                             'Month':'build_month_{}'.format(statu),'Wday':'build_Wday_{}'.format(statu),

                             'Diff_days':'build_daydiff_{}'.format(statu),'room_size':'build_room_{}'.format(statu)},inplace=True)

        subdata=pd.merge(subdata,temp,on='building_id',how='left')



    t7=data[fea].reset_index(drop=True)

    t7=pd.concat((t7,data1[['Month','Wday','Diff_days']]),axis=1)

    t7['room_size']=t7['bathrooms']+t7['bedrooms']

    t7.drop(['bathrooms','bedrooms'],axis=1,inplace=True)



    t8=t7[['building_id','manager_id']]

    t8['build_manager_count']=1

    t8=t8.groupby(['building_id','manager_id']).agg('sum').reset_index()

    subdata=pd.merge(subdata,t8,on=['building_id','manager_id'],how='left')

    

    for i,statu in enumerate(status):

        temp=t7.groupby(['building_id','manager_id']).agg(statu).reset_index()

        temp.rename(columns={'price':'bm_price_{}'.format(statu),'latitude':'bm_lat_{}'.format(statu),'longitude':'bm_lon_{}'.format(statu),

                             'Month':'bm_month_{}'.format(statu),'Wday':'bm_Wday_{}'.format(statu),

                       'Diff_days':'bm_daydiff_{}'.format(statu),'room_size':'bm_room_{}'.format(statu)},inplace=True)

        subdata=pd.merge(subdata,temp,on=['building_id','manager_id'],how='left')



    subdata['build_type']=subdata['building_id'].apply(lambda x : 0 if x==0 else 1)

    subdata.drop(['building_id','manager_id','Month','Wday','Diff_days'],axis=1,inplace=True)

    return subdata
def Day_situation(x):

    if x<=10:

        return 1

    elif x>20:

        return 3

    else:

        return 2

        

def process_created(data):

    subdata=pd.DataFrame(data['created'])

    subdata['date']=pd.to_datetime(data['created'])

    subdata['Month']=subdata.date.dt.month

    subdata['Season']=1

    subdata.loc[subdata['Month'].isin([4,5,6]),'Season']=2

    subdata.loc[subdata['Month'].isin([7,8,9]),'Season']=3

    subdata.loc[subdata['Month'].isin([10,11,12]),'Season']=4

    subdata['Day']=subdata.date.dt.day

    subdata['day_situation']=subdata.Day.apply(Day_situation)

    subdata['Wday']=subdata.date.dt.dayofweek

    subdata['isWeekday']=subdata.Wday.apply(lambda x: 1 if x in [0,6] else 0)

    subdata['Yday']=subdata.date.dt.dayofyear

    subdata['Hour']=subdata.date.dt.hour

    subdata['isNight']=subdata.Hour.apply(lambda x:1 if x>12 else 0)

    subdata['Diff_days']=subdata['created'].apply(lambda x:(datetime.date(2017,3,1)-

    datetime.date(int(x.split(' ')[0].split('-')[0]),int(x.split(' ')[0].split('-')[1]),int(x.split(' ')[0].split('-')[2]))).days)



    subdata.drop(['created','date'],axis=1,inplace=True)

    subdata=subdata.reset_index(drop=True)

    return subdata
def process_managerid(data,data1):

    fea=['manager_id','price','bathrooms','bedrooms','latitude','longitude']

    subdata=data[fea].reset_index(drop=True)

    subdata=pd.concat((subdata,data1[['Month','Wday','Diff_days']]),axis=1)

    t=subdata[['manager_id','price']]

    t['manager_count']=1

    t=t.groupby(['manager_id']).agg('count').reset_index()

    t.drop('price',axis=1,inplace=True)



    t1=subdata[['manager_id','price','bathrooms','bedrooms','latitude','longitude','Month','Wday','Diff_days']]

    t1['room_size']=t1['bathrooms']+t1['bedrooms']

    t1.drop(['bathrooms','bedrooms'],axis=1,inplace=True)

    subdata=pd.merge(subdata,t,on='manager_id',how='left')

    

    status=['sum','mean','min','max','median']

    for statu in status:

        temp=t1.groupby('manager_id').agg(statu).reset_index()

        temp.rename(columns={'price':'manager_price_{}'.format(statu),'latitude':'manager_lat_{}'.format(statu),'longitude':'manager_lon_{}'.format(statu),

                   'Month':'manager_month_{}'.format(statu),'Wday':'manager_Wday_{}'.format(statu),

                   'Diff_days':'manager_daydiff_{}'.format(statu),'room_size':'manager_room_{}'.format(statu)},inplace=True)

        subdata=pd.merge(subdata,temp,on='manager_id',how='left')

    subdata.drop(['manager_id','price','bathrooms','bedrooms','latitude','longitude','Month','Wday','Diff_days'],axis=1,inplace=True)

    return subdata

def get_word_feature(data):

    fea=['description','display_address','features','photos','street_address']

    subdata=data[fea].reset_index(drop=True)

    feature_transform = CountVectorizer(stop_words='english', max_features=150)

    #####features########

    subdata['features'] = subdata["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))

#    data1['features'] = data1["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))

    feature_transform.fit(list(subdata['features']))

    vocabulary=feature_transform.vocabulary_

    feat_sparse = feature_transform.transform(subdata["features"])

    fea_counter = pd.DataFrame([pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0])])

    fea_counter.columns = list(sorted(vocabulary.keys()))

    subdata['features_count']=subdata['features'].apply(lambda x:len(x))

    #####description######

    subdata['description'] = subdata['description'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))

    subdata['description'] = subdata['description'].apply(lambda x: x.replace('!<br /><br />', ''))



    string.punctuation.__add__('!!')

    string.punctuation.__add__('(')

    string.punctuation.__add__(')')

    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))



    subdata['description'] = subdata['description'].apply(lambda x: x.translate(remove_punct_map))

    subdata['desc_letter_count']=subdata['description'].apply(lambda x:len(x.strip()))

    subdata['desc_words_count'] = subdata['description'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

    ######adddress########

    subdata['address1'] = subdata['display_address']

    subdata['address1'] = subdata['address1'].apply(lambda x: x.lower())



    address_map = {

        'w': 'west','st.': 'street','ave': 'avenue',

        'st': 'street','e': 'east','n': 'north','s': 'south'}



    def address_map_func(s):

        s = s.split(' ')

        out = []

        for x in s:

            if x in address_map:

                out.append(address_map[x])

            else:

                out.append(x)

        return ' '.join(out)



    subdata['address1'] = subdata['address1'].apply(lambda x: x.translate(remove_punct_map))

    subdata['address1'] = subdata['address1'].apply(lambda x: address_map_func(x))

    new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']



    for col in new_cols:

        subdata[col] = subdata['address1'].apply(lambda x: 1 if col in x else 0)

    subdata['other_address'] = subdata[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

    ###########photos#################

    subdata['photos_count'] = subdata['photos'].apply(lambda x: len(str(x).split(',')))

    subdata.drop(['description','display_address','features','photos','street_address','address1'],axis=1,inplace=True)



    subdata=pd.concat([subdata,fea_counter],axis=1)

    return subdata
train,test,y,listing_id=load_data()

train.drop('interest_level',axis=1,inplace=True)

n=train.shape[0]

data=pd.concat([train,test])



data1=add_room_price(data)

data2=process_created(data) 

data3=process_buildingid(data,data2)

data4=process_managerid(data,data2) 

data5=get_word_feature(data)

data_concat=pd.concat([data1,data2,data3,data4,data5],axis=1)



X_train=data_concat.iloc[:n,:]

X_test=data_concat.iloc[n:,:]



X_training,X_val,y_training,y_val=train_test_split(X_train,y,test_size=0.3,random_state=0)

dtrain=xgb.DMatrix(data=X_training,label=y_training)

dval=xgb.DMatrix(data=X_val,label=y_val)

dtest=xgb.DMatrix(data=X_test)



params = {

    'eta':.02,

    'max_depth':4,

    'min_child_weight':3,

    "n_estimators":600,

    'early_stopping_rounds':30,

    'colsample_bytree':.7,

    'subsample':.7,

    'gamma':0.1,

    'seed':0,

    'nthread':-1,

    'objective':'multi:softprob',

    'eval_metric':'mlogloss',

    'num_class':3,

    'silent':1

    }



watchlist=[(dtrain,'train'),(dval,'val')]

#xgb_cv=xgb.cv(params,dtrain, num_boost_round=5600, nfold=4,seed=0)

#print "Min_logloss{}".format(min(xgb_cv['test-mlogloss-mean']))

#print "Best_Rounds{}".format(np.argmin(xgb_cv['test-mlogloss-mean']))

#best_rounds = np.argmin(xgb_cv['test-mlogloss-mean'])

bst=xgb.train(params,dtrain,3500,evals=watchlist) #cv 0.5281

pre_test=bst.predict(dtest)



def prepare_submission(preds):

    now = datetime.datetime.now()   

    submission = pd.DataFrame(data = {'listing_id': listing_id})

    submission['low'] = preds[:, 0]

    submission['medium'] = preds[:, 1]

    submission['high'] = preds[:, 2]

    sub_file = './Submission/'+'Submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    submission.to_csv(sub_file,index = False)

prepare_submission(pre_test)