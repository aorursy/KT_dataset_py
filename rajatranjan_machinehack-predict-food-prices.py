# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/participants_data_final/Participants_Data_Final"))



# Any results you write to the current directory are saved as output.
train=pd.read_excel('../input/participants_data_final/Participants_Data_Final/Data_Train.xlsx')

test=pd.read_excel('../input/participants_data_final/Participants_Data_Final/Data_Test.xlsx')

s=pd.read_excel('../input/participants_data_final/Participants_Data_Final/Sample_submission.xlsx')

train.head()
train.describe(include='all').T
train[train['RESTAURANT_ID']==6571]
train[train.duplicated()==True]
train[train['RESTAURANT_ID'].isin(test['RESTAURANT_ID'])]
test.describe(include='all').T
train[train['RATING']=='NEW']['TITLE'].value_counts()
train[train['TITLE']=='QUICK BITES']['COST'].describe()
train[train['VOTES'].isnull()==True].describe(include='all')
train[train['CITY']=='Kochi']['RATING'].value_counts()
train.groupby('TITLE')['COST'].describe().sort_values('mean',ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(12,10))

sns.boxplot(train['COST'])
train.iloc[246,4]='Kochi'

train.iloc[817,4]='Mumbai'

train.iloc[5149,4]='Navi Mumbai'

train.iloc[5297,4]='Kochi'

train.iloc[6397,4]='Mumbai'

train.iloc[6451,4]='Chennai'

train.iloc[8456,4]='Bangalore'

train.iloc[8735,4]='Chennai'

train.iloc[9121,4]='Bangalore'

train.iloc[9268,4]='Kochi'

train.iloc[10200,4]='Mumbai'

train.iloc[10704,4]='Chennai'

train.iloc[11491,4]='Chennai'

train.iloc[12314,4]='Hyderabad'

train.iloc[12447,4]='Hyderabad'



test.iloc[169,4]='Chennai'

test.iloc[710,4]='Chennai'

test.iloc[1354,4]='Uttar Pradesh'

test.iloc[3621,4]='Mumbai'

test.iloc[4167,4]='Chennai'



train.iloc[1365,0]='Kerela'

train=train[train.duplicated()==False]

df=train.append(test,ignore_index=True)

df.head()
# Feature Engineering

train['COST'].describe()
train.isnull().sum()
train['RATING'].unique()
train.isnull().sum()
import re

def timeclean(v):

    return re.sub("[^a-zA-Z0-9:]", " ",v)

df['TITLE']=df['TITLE'].apply( lambda x :" ".join(x.split(",")))

df['CUISINES']=df['CUISINES'].apply( lambda x :" ".join(x.split(",")))

df['CITY'].fillna('NOTFOUND',inplace=True)

df['LOCALITY'].fillna('NOTFOUND',inplace=True)

df['LOCATION']=df['CITY']+' '+df['LOCALITY']

df['CITY']=df['CITY'].apply(lambda x : re.sub("[^0-9a-zA-Z]", " ",x))

df['Serves2Time']=df['TIME'].apply(lambda x :1 if len(x.split(','))>1 else 0)

df['TIME']=df['TIME'].apply(timeclean)

df['RATING'].replace({'NEW': '3.7',np.nan:'0.0','-':'3.7'},inplace=True)

df['RATING']=df['RATING'].astype(np.float64)

df['RATING.BINS']=pd.cut(df['RATING'],4,labels=['Sly','Fair','Good','Great']).astype(np.object)



df['VOTES'].fillna('0 votes',inplace=True)

df['VOTES']=df['VOTES'].apply(lambda x : int(x.split(' ')[0]))

df['VOTES.BINS']=pd.cut(df['VOTES'],5,labels=['Sly','Fair','Good','Great','Awesome']).astype(np.object)

df=pd.get_dummies(df,columns=['RATING.BINS','VOTES.BINS'],drop_first=True)



df.drop(['RESTAURANT_ID'],axis=1,inplace=True)

df.head()
df=df.merge(df.groupby('TITLE')['RATING'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'trating_mean','sum':'trating_sum',

                                                                                       'min':'trating_min','max':'trating_max','quantile':'trating_quant'}).reset_index(),on='TITLE',how='left')



df=df.merge(df.groupby('TITLE')['VOTES'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'tVOTES_mean','sum':'tVOTES_sum',

                                                                                       'min':'tVOTES_min','max':'tVOTES_max','quantile':'tVOTES_quant'}).reset_index(),on='TITLE',how='left')



df=df.merge(df.groupby('CUISINES')['VOTES'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'cVOTES_mean','sum':'cVOTES_sum',

                                                                                       'min':'cVOTES_min','max':'cVOTES_max','quantile':'cVOTES_quant'}).reset_index(),on='CUISINES',how='left')

df=df.merge(df.groupby('CUISINES')['RATING'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'cRATING_mean','sum':'cRATING_sum',

                                                                                       'min':'cRATING_min','max':'cRATING_max','quantile':'cRATING_quant'}).reset_index(),on='CUISINES',how='left')





df=df.merge(df.groupby('CITY')['RATING'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'CITYrating_mean','sum':'CITYrating_sum',

                                                                                       'min':'CITYrating_min','max':'CITYrating_max','quantile':'CITYrating_quant'}).reset_index(),on='CITY',how='left')



df=df.merge(df.groupby('CITY')['VOTES'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'CITYVOTES_mean','sum':'CITYVOTES_sum',

                                                                                       'min':'CITYVOTES_min','max':'CITYVOTES_max','quantile':'CITYVOTES_quant'}).reset_index(),on='CITY',how='left')



df=df.merge(df.groupby('LOCALITY')['VOTES'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'LOCALITYVOTES_mean','sum':'LOCALITYVOTES_sum',

                                                                                       'min':'LOCALITYVOTES_min','max':'LOCALITYVOTES_max','quantile':'LOCALITYVOTES_quant'}).reset_index(),on='LOCALITY',how='left')

df=df.merge(df.groupby('LOCALITY')['RATING'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'LOCALITYRATING_mean','sum':'LOCALITYRATING_sum',

                                                                                       'min':'LOCALITYRATING_min','max':'LOCALITYRATING_max','quantile':'LOCALITYRATING_quant'}).reset_index(),on='LOCALITY',how='left')

df=df.merge(df.groupby('TIME')['VOTES'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'TIMEVOTES_mean','sum':'TIMEVOTES_sum',

                                                                                       'min':'TIMEVOTES_min','max':'TIMEVOTES_max','quantile':'TIMEVOTES_quant'}).reset_index(),on='TIME',how='left')

df=df.merge(df.groupby('TIME')['RATING'].agg(['mean','sum','min','max','quantile']).rename(columns={'mean':'TIMERATING_mean','sum':'TIMERATING_sum',

                                                                                       'min':'TIMERATING_min','max':'TIMERATING_max','quantile':'TIMERATING_quant'}).reset_index(),on='TIME',how='left')
df.head()
df_train = df[df['COST'].isnull()==False]

df_test = df[df['COST'].isnull()==True]

print(df_train.shape,df_test.shape)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

v_city = TfidfVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

city_tr =v_city.fit_transform(df_train['CITY'])

city_ts =v_city.transform(df_test['CITY'])



c_city = CountVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

ccity_tr =c_city.fit_transform(df_train['CITY'])

ccity_ts =c_city.transform(df_test['CITY'])



v_cuisine = TfidfVectorizer(ngram_range=(1,1),stop_words="english", analyzer='word')

cui_tr =v_cuisine.fit_transform(df_train['CUISINES'])

cui_ts =v_cuisine.transform(df_test['CUISINES'])



v_local = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

local_tr =v_local.fit_transform(df_train['LOCALITY'])

local_ts =v_local.transform(df_test['LOCALITY'])



v_time = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word')

time_tr =v_time.fit_transform(df_train['TIME'])

time_ts =v_time.transform(df_test['TIME'])



v_title = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word')

title_tr =v_title.fit_transform(df_train['TITLE'])

title_ts =v_title.transform(df_test['TITLE'])



v_loc = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

loc_tr =v_loc.fit_transform(df_train['LOCATION'])

loc_ts =v_loc.transform(df_test['LOCATION'])
df.columns


col=[ 'RATING', 

       'VOTES', 'RATING.BINS_Good',

       'RATING.BINS_Great', 'RATING.BINS_Sly', 'VOTES.BINS_Fair',

       'VOTES.BINS_Good', 'VOTES.BINS_Sly', 'trating_mean', 'trating_sum',

       'trating_min', 'trating_max', 'trating_quant', 'tVOTES_mean',

       'tVOTES_sum', 'tVOTES_min', 'tVOTES_max', 'tVOTES_quant', 'cVOTES_mean',

       'cVOTES_sum', 'cVOTES_min', 'cVOTES_max', 'cVOTES_quant',

       'cRATING_mean', 'cRATING_sum', 'cRATING_min', 'cRATING_max',

       'cRATING_quant', 'CITYrating_mean', 'CITYrating_sum', 'CITYrating_min',

       'CITYrating_max', 'CITYrating_quant', 'CITYVOTES_mean', 'CITYVOTES_sum',

       'CITYVOTES_min', 'CITYVOTES_max', 'CITYVOTES_quant',

       'LOCALITYVOTES_mean', 'LOCALITYVOTES_sum', 'LOCALITYVOTES_min',

       'LOCALITYVOTES_max', 'LOCALITYVOTES_quant', 'LOCALITYRATING_mean',

       'LOCALITYRATING_sum', 'LOCALITYRATING_min', 'LOCALITYRATING_max',

       'LOCALITYRATING_quant', 'TIMEVOTES_mean', 'TIMEVOTES_sum',

       'TIMEVOTES_min', 'TIMEVOTES_max', 'TIMEVOTES_quant', 'TIMERATING_mean',

       'TIMERATING_sum', 'TIMERATING_min', 'TIMERATING_max',

       'TIMERATING_quant']

df_train[col].head()
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((df_train[col],city_tr,cui_tr,local_tr,loc_tr,time_tr,title_tr,  ccity_tr )).tocsr()

final_featurest = sparse.hstack((df_test[col],city_ts,cui_ts,local_ts,loc_ts,time_ts,title_ts,  ccity_ts )).tocsr()
from sklearn.model_selection import train_test_split

import math

from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,mean_squared_log_error

X=final_features

y=np.log1p(df_train['COST'].astype(np.float64))

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state = 1994)
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,BayesianRidge

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.metrics import mean_squared_log_error

import math

def rmsle(real, predicted):

    real=np.expm1(real)

    predicted=np.expm1(predicted)

    return np.sqrt(mean_squared_log_error(real,predicted))

    

def rmsle_lgb(labels, preds):

    return 'rmsle', rmsle(preds,labels), False
m=LGBMRegressor(n_estimators=4000,random_state=1994,learning_rate=0.01,objective='regression',reg_alpha=1,reg_lambda=10,colsample_bytree=0.2,min_child_samples=20,feature_fraction=0.2)

m.fit(X_train,y_train,eval_set=[(X_val, y_val.values)],eval_metric='rmse', early_stopping_rounds=100,verbose=100)

p=m.predict(X_val)

print(rmsle_lgb(y_val.values,p))
from xgboost import XGBRegressor

m=XGBRegressor(n_estimators=6000,learning_rate=0.02,random_state=1994,max_depth=8,reg_alpha=1,colsample_bytree=0.3,max_delta_step=0.5,seed=1994,colsample_bylevel=0.5)

m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val, y_val.values)],eval_metric='rmse', early_stopping_rounds=200,verbose=100)

p1=m.predict(X_val)

print(rmsle_lgb(y_val.values,p1))
print(rmsle_lgb(y_val,(p*0.5+p1*0.5)))
errlgb=[]

y_pred_totlgb=[]

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=20,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 4000,

                   'n_jobs': -1,'learning_rate':0.01,'random_state':1994,'reg_lambda':10,'reg_alpha':1,'colsample_bytree':0.2

                  ,'min_child_samples':20,'feature_fraction':0.2}

    rf=LGBMRegressor(**lgbm_params)

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

         eval_metric=rmsle_lgb,

         verbose=200

         , early_stopping_rounds=200

          )

    pr=rf.predict(X_test)

    print("errlgb: ",rmsle_lgb(y_test.values,pr)[1])

    

    errlgb.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(final_featurest)

    y_pred_totlgb.append(p)
errxgb=[]

y_pred_totxgb=[]

i=0

from sklearn.model_selection import KFold,StratifiedKFold

fold=KFold(n_splits=20,shuffle=True,random_state=1994)

for train_index, test_index in fold.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    lgbm_params = {'n_estimators': 6000,

                   'n_jobs': -1,'learning_rate':0.02,'random_state':1994,'max_depth':8,'reg_alpha':1,'colsample_bytree':0.3

                  ,'max_delta_step':0.5,'colsample_bylevel':0.5,'seed':1994}

    rf=XGBRegressor(**lgbm_params)

    rf.fit(X_train,y_train,eval_set=[(X_test, y_test.values)],

         eval_metric='rmse',

         verbose=200

         , early_stopping_rounds=100

          )

    pr=rf.predict(X_test)

    print("errlgb: ",rmsle_lgb(y_test.values,pr)[1])

    

    errxgb.append(rmsle_lgb(y_test.values,pr)[1])

    p = rf.predict(final_featurest)

    y_pred_totxgb.append(p)
np.mean(errxgb),np.mean(errlgb)
np.mean(errxgb)*0.5+np.mean(errlgb)*0.5
np.mean(y_pred_totlgb,0)*0.5+np.mean(y_pred_totxgb,0)*0.5
s=pd.DataFrame({'COST':np.expm1(np.mean(y_pred_totlgb,0)*0.5+np.mean(y_pred_totxgb,0)*0.5)})

s.to_excel('MH-Predict_food_pricesv9_stack_final.xlsx',index=False)

s.head()



s=pd.DataFrame({'COST':np.expm1(np.mean(y_pred_totlgb,0))})

s.to_excel('MH-Predict_food_pricesv9_lgb_final.xlsx',index=False)

s.head()



s=pd.DataFrame({'COST':np.expm1(np.mean(y_pred_totxgb,0))})

s.to_excel('MH-Predict_food_pricesv9_xgb_final.xlsx',index=False)

s.head()