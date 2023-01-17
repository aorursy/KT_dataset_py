from sklearn.model_selection import train_test_split,KFold

from sklearn.model_selection import train_test_split

from xgboost import plot_tree

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pdb

import lightgbm as lgb

from sklearn import model_selection, preprocessing

import xgboost as xgb

import matplotlib.pyplot as plt

import datetime

#now = datetime.datetime.now()

from collections import Counter

import pickle



def prepare_data():

    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])



    # In the actual script, I used these scores which can be produced from

    # kernels:

    # https://www.kaggle.com/schoolpal/nn-model-lb-0-306-to-0-308

    # https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script

    # (xgb_train,xgb_test)=pickle.load(open('xgb_predicted.pkl'))

    # (nn_train,nn_test)=pickle.load(open('nn_predicted.pkl'))

    # (lgb_train,lgb_test)=pickle.load(open('lgb_predicted.pkl'))

    # train['nn_score']=nn_train

    # train['nn_score_log']=np.log(nn_train)

    # train['xgb_score']=xgb_train

    # train['xgb_score_log']=np.log(xgb_train)

    

    # test['nn_score']=nn_test

    # test['nn_score_log']=np.log(nn_test)

    # test['xgb_score']=xgb_test

    # test['xgb_score_log']=np.log(xgb_test)

    

    train = train[(train.product_type=='Investment') & (train.timestamp>pd.to_datetime('2013-01-01'))]

    test = test[(test.product_type=='Investment')]

    id_test = test.id

    full_sq=train.full_sq.copy()

    full_sq[full_sq<5]=np.NaN

    price_sq=train.price_doc/full_sq

    y_train = ((train["price_doc"]<=1e6) |  (train["price_doc"]==2e6) |  (train["price_doc"]==3e6) | (price_sq<30000) ).astype(int)

    #can't merge train with test because the kernel run for very long time



    num_train=len(train)

    times=pd.concat([train.timestamp,test.timestamp])

    x_train = train.drop(['id','timestamp','price_doc','product_type'], axis=1)

    x_test = test.drop(['id','timestamp', 'product_type'], axis=1)

    df_all=pd.concat([x_train,x_test])

    df_all['olds']=times.dt.year-df_all.build_year

    da=df_all

    to_remove=[]

    df_cat=None

    for c in da.columns:

        if da[c].dtype=='object':

            oh=pd.get_dummies(da[c],prefix=c)

            

            if df_cat is None:

                df_cat=oh

            else:

                df_cat=pd.concat([df_cat,oh],axis=1)

            to_remove.append(c)

    da.drop(to_remove,inplace=True,axis=1)

    to_remove=[]

    if df_cat is not None:

        sums=df_cat.sum(axis=0)

        to_remove=sums[sums<200].index.values

        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]

        da = pd.concat([da, df_cat], axis=1)





    x_train=df_all[:len(x_train)]

    x_test=df_all[len(x_train):]

    return x_train,x_test,y_train,times





def model(x_train,y_train,x_test):

    RS=1

    np.random.seed(RS)

    ROUNDS = 1500

    params = {

        'objective': 'binary',

            'boosting': 'gbdt',

            'learning_rate': 0.01 ,

            'verbose': 0,

            'num_leaves': 2 ** 5,

            'bagging_fraction': 0.95,

            'bagging_freq': 1,

            'bagging_seed': RS,

            'feature_fraction': 0.7,

            'feature_fraction_seed': RS,

            'max_bin': 100,

            'max_depth': 7,

            'num_rounds': ROUNDS,

        }

    train_lgb=lgb.Dataset(x_train,y_train)

    model=lgb.train(params,train_lgb,num_boost_round=ROUNDS)

    predict=model.predict(x_test)

    return predict

def precision(predict,gold):

    correct=np.count_nonzero((predict==1) & (gold==1))

    p=float(correct)/np.count_nonzero(predict)

    return p,correct

x_train,x_test,y_train,times=prepare_data()



# 5-fold cross validation

skf = KFold(5,shuffle=False,random_state=1)

p_mean=0

top10=[]

top30=[]

for i,(train_inds,test_inds) in enumerate(skf.split(x_train)):

    print('Working on CV '+str(i))

    val=x_train.iloc[test_inds]

    y_val=y_train.iloc[test_inds]

    cv_train=x_train.iloc[train_inds]

    start_time=times.iloc[test_inds[0]]

    cv_y_train=y_train.iloc[train_inds]

    y_predict=model(cv_train,cv_y_train,val)

    sorted_inds=np.argsort(y_predict)[::-1]

    labels=np.zeros(len(y_predict))

    labels[sorted_inds[0:10]]=1

    p,count=precision(labels,y_val)

    top10.append((p,start_time))

#    p_mean+=p

#    print(str(i)+'. TOP 10 precision '+str(p)+' '+str(start_time))

    labels=np.zeros(len(y_predict))

    labels[sorted_inds[0:30]]=1

    p,count=precision(labels,y_val)

    top30.append((p,start_time))



for i,r in enumerate(top10):

    print('Fold '+str(i)+'. TOP 10 precision '+str(r[0])+' '+str(r[1]))

    

for i,r in enumerate(top30):

    print('Fold '+str(i)+'. TOP 30 precision '+str(r[0])+' '+str(r[1]))


