import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os 

import seaborn as sns

import geopandas as gpd

import folium

from folium import plugins

import datetime

import math
os.listdir("../input/ieee-fraud-detection")
train_identity=pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_transaction=pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

test_transaction=pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

test_identity=pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

print("train_identity_data_size: ",len(train_identity))

print("train_transaction_data_size: ",len(train_transaction))

print("test_transaction_data_size: ",len(test_transaction))

print("test_identity_data_size: ",len(test_identity))
train_identity.head()
plt.figure(figsize=(10,10))

sns.barplot(x=train_identity.isnull().sum().sort_values(ascending=False),y=train_identity.isnull().sum().sort_values(ascending=False).index)

plt.title("counts of missing value for train_identity",size=20)
train_identity_new=pd.DataFrame(train_identity,columns=['TransactionID','id_01','id_12','id_38','id_37','id_36','id_35','id_15','id_29','id_28','id_11','id_02','DeviceType','id_31','id_17','id_19','id_20'])

train_identity_new=train_identity_new.dropna(subset=['id_38','id_37','id_36','id_35','id_15','id_29','id_28','id_11','id_02','DeviceType','id_31','id_17','id_19','id_20'])

train_identity_new.head()
len(train_identity_new)
plt.figure(figsize=(8,8))

corr = train_identity_new.corr()

sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,annot=True)

plt.title("correlation plot for train_identity_new",size=28)
fig,ax=plt.subplots(2,2,figsize=(20,20))

y=train_identity_new.id_01.value_counts().index

x=train_identity_new.id_01.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,0],orient='h')

ax[0,0].set_title("Bar chart for id_01",size=20)

ax[0,0].set_xlabel('counts',size=18)

ax[0,0].set_ylabel('')



y=train_identity_new.id_12.value_counts().index

x=train_identity_new.id_12.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,1])

ax[0,1].set_title("Bar chart for id_12",size=20)

ax[0,1].set_xlabel('counts',size=18)

ax[0,1].set_ylabel('')



y=train_identity_new.id_38.value_counts().index

x=train_identity_new.id_38.value_counts()

sns.barplot(x=x,y=y,ax=ax[1,0],order=['T','F'])

ax[1,0].set_title("Bar chart for id_38",size=20)

ax[1,0].set_xlabel('counts',size=18)

ax[1,0].set_ylabel('')



y=train_identity_new.id_37.value_counts().index

x=train_identity_new.id_37.value_counts()

sns.barplot(x=x,y=y,ax=ax[1,1],order=['T','F'])

ax[1,1].set_title("Bar chart for id_37",size=20)

ax[1,1].set_xlabel('counts',size=18)

ax[1,1].set_ylabel('')
fig,ax=plt.subplots(2,2,figsize=(20,20))

y=train_identity_new.id_36.value_counts().index

x=train_identity_new.id_36.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,0],order=['T','F'])

ax[0,0].set_title("Bar chart for id_36",size=20)

ax[0,0].set_xlabel('counts',size=18)

ax[0,0].set_ylabel('')



y=train_identity_new.id_35.value_counts().index

x=train_identity_new.id_35.value_counts()

sns.barplot(x=x,y=y,ax=ax[0,1],order=['T','F'])

ax[0,1].set_title("Bar chart for id_35",size=20)

ax[0,1].set_xlabel('counts',size=18)

ax[0,1].set_ylabel('')



y=train_identity_new.id_15.value_counts().index

x=train_identity_new.id_15.value_counts()

sns.barplot(x=x,y=y,ax=ax[1,0])

ax[1,0].set_title("Bar chart for id_15",size=20)

ax[1,0].set_xlabel('counts',size=18)

ax[1,0].set_ylabel('')



y=train_identity_new.id_29.value_counts().index

x=train_identity_new.id_29.value_counts()

sns.barplot(x=x,y=y,ax=ax[1,1])

ax[1,1].set_title("Bar chart for id_29",size=20)

ax[1,1].set_xlabel('counts',size=18)

ax[1,1].set_ylabel('')
del train_identity

del test_identity

train_transaction.head()
len(train_transaction)
plt.figure(figsize=(18,70))

sns.barplot(x=train_transaction.isnull().sum().sort_values(ascending=False),y=train_transaction.isnull().sum().sort_values(ascending=False).index)

plt.title("counts of missing value for train_transaction",size=20)


train_transaction_new=pd.DataFrame(train_transaction,columns=train_transaction.isnull().sum().sort_values()[:250].index)

train_transaction_new=train_transaction_new.drop(columns=['TransactionID'])

train_transaction_new_label=train_transaction_new.isFraud

train_transaction_new=train_transaction_new.drop(columns=['isFraud'])

train_transaction_new.head()
len(train_transaction_new)
test_transaction.head()
len(test_transaction)
plt.figure(figsize=(18,70))

sns.barplot(x=test_transaction.isnull().sum().sort_values(ascending=False),y=test_transaction.isnull().sum().sort_values(ascending=False).index)

plt.title("counts of missing value for test_transaction",size=20)
test_transaction_new=pd.DataFrame(test_transaction,columns=train_transaction.isnull().sum().sort_values()[:250].index)

del test_transaction

del train_transaction

ID=test_transaction_new.TransactionID

test_transaction_new=test_transaction_new.drop(columns=['TransactionID','isFraud'])

test_transaction_new.head()
len(test_transaction_new)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for i in list(train_transaction_new.select_dtypes(include=['object']).columns):

    test_transaction_new[i] = labelencoder.fit_transform(test_transaction_new[i].astype('str'))

    train_transaction_new[i] = labelencoder.fit_transform(train_transaction_new[i].astype('str'))

test_transaction_new.ProductCD[:5]

train_transaction_new.ProductCD[:5]
#train_transaction_new=train_transaction_new.fillna(-999)

#test_transaction_new=test_transaction_new.fillna(-999)

train_transaction_new=train_transaction_new.fillna(train_transaction_new.median())

test_transaction_new=test_transaction_new.fillna(train_transaction_new.median())
from sklearn.linear_model import LogisticRegression  

from sklearn.preprocessing import StandardScaler  

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_transaction_new,train_transaction_new_label,test_size=0.2)

del train_transaction_new

lr = LogisticRegression(C=0.09,solver='lbfgs')  

lr.fit(X_train, y_train)  

proba_test = lr.predict_proba(X_test)[:, 1]

LR_result=pd.DataFrame({'pred':proba_test,'real':y_test})

LR_result['pred_0_1']=LR_result.pred.apply(lambda x:1 if x>=0.5 else 0)
print('LR_acc: ',sum(LR_result.real==LR_result.pred_0_1)/len(LR_result))
import lightgbm as lgb  

import pickle  

from sklearn.metrics import roc_auc_score  

lgb_train = lgb.Dataset(X_train, y_train)  

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train) 

params = {  

    'boosting_type': 'gbdt',  

    'objective': 'binary',  

    'metric': {'binary_logloss', 'auc'},  

    'num_leaves':240,  

    'max_depth': 15,  

    'min_data_in_leaf': 100,  

    'learning_rate': 0.05,  

    'feature_fraction': 0.95,  

    'bagging_fraction': 0.95,  

    'bagging_freq': 5,  

    'lambda_l1': 0,    

    'lambda_l2': 0, 

    'min_gain_to_split': 0.1,  

    'verbose': 0,  

    'is_unbalance': True  

}  
gbm = lgb.train(params,  lgb_train,  

                num_boost_round=10000,  

                valid_sets=lgb_eval,early_stopping_rounds=500)  
gbm.predict(test_transaction_new[:10], num_iteration=gbm.best_iteration) 
LR_TEST=lr.predict_proba(test_transaction_new)[:, 1]

LGBM_TEST= gbm.predict(test_transaction_new, num_iteration=gbm.best_iteration) 



prediction=pd.DataFrame({'TransactionID':ID,'LR_TEST':LR_TEST,'LGBM_TEST':LGBM_TEST})



prediction.to_csv('prediction.csv',index=False)



submission=pd.DataFrame({'TransactionID':ID,'isFraud':LGBM_TEST})





submission.to_csv('submission.csv',index=False)