#Import required libraries

import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train_sample.csv')

df.head()
#convert timestamp to datatime

def todatetime(df):

    df['click_time']=pd.to_datetime(df['click_time'])

    df['click_hour']=df['click_time'].dt.hour

    df['click_day']=df['click_time'].dt.day

    df['click_weekday']=df['click_time'].dt.weekday

    df['click_month']=df['click_time'].dt.month

    df['click_year']=df['click_time'].dt.year

    return df
df=todatetime(df)
df=df.drop(['click_time'],axis=1)
df.head()
df.isnull().sum()
df=df.drop('attributed_time',axis=1)
#Shuffling observations

df=df.sample(frac=1)

df
import seaborn as sn

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

sn.countplot(x='is_attributed',data=df)

plt.ylabel('Number of clicks')

plt.show()
df['is_attributed'].value_counts()
target_label=df['is_attributed']

target_label.shape
ones=df[df['is_attributed']==1]

zeros=df[df['is_attributed']==0]
df=df.drop(['is_attributed','ip'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df,target_label,test_size=0.2,random_state=42)

print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=42)

print(x_train.shape,y_train.shape)

print(x_val.shape,y_val.shape)
import lightgbm as lgb

#load datasets in lgb formate

train_data=lgb.Dataset(x_train,label=y_train,free_raw_data=False)

validation_data=lgb.Dataset(x_val,label=y_val,free_raw_data=False)
#set parameters for training

params={ 'num_leaves':160,

        'object':'binary',

        'metric':['auc','binary_logloss']

       }


#Original LGB model before sampling

num_round=100

def lgb_basemodel(x_train,y_train):

    lgb_model=lgb.train(params,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=20)

    return lgb_model
#LGBM model after resampling the data using Under sampling techniques

from imblearn.under_sampling import RandomUnderSampler 

def lgb_downsampling(x_train,y_train):

    lgb_enn=RandomUnderSampler(random_state=42)

    x_resample,y_resample=lgb_enn.fit_resample(x_train,y_train)

    train_data=lgb.Dataset(x_resample,label=y_resample,free_raw_data=False)

    lgb_model=lgb.train(params,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=20)

    return lgb_model,x_resample,y_resample;
#LGBM model after resampling the data using Up sampling techniques

from imblearn.over_sampling import SMOTE  #Balances the classes by performing upsampling on minority class

def lgb_upsampling(x_train,y_train):

    lgb_smote= SMOTE(random_state=42)

    x_resample,y_resample=lgb_smote.fit_resample(x_train,y_train)

    train_data=lgb.Dataset(x_resample,label=y_resample)

    lgb_model=lgb.train(params,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=20)

    return lgb_model,x_resample,y_resample;
weight_factor=zeros.shape[0]/ones.shape[0]  # Ratio of number of samples in majority class to number of samples in minority class

print('Weight factor is %0.2f'%(weight_factor))
#set parameters for training

params1={ 'num_leaves':160,

        'object':'binary',

        'metric':['auc','binary_logloss'],

        'scale_pos_weight':397.41                 #Weight of minority class

       }
#LGBM model using Upweighting technique for handling the imbalanced data

def lgb_Upweighting(x_train,y_train):

    lgb_model=lgb.train(params1,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=20)

    return lgb_model;
#Basemodel

lgb_basemodel=lgb_basemodel(x_train,y_train)
#Downsampling model

lgb_downsampling,x_down,y_down=lgb_downsampling(x_train,y_train)
y_down_df=pd.DataFrame(y_down)

y_down.shape
plt.hist(y_down);
#Upsampling model

lgb_upsampling,x_up,y_up=lgb_upsampling(x_train,y_train)
y_up_df=pd.DataFrame(y_up)

y_up.shape
plt.hist(y_up);
#Upweighting model 

lgb_upweighting=lgb_Upweighting(x_train,y_train);
#Basemodel

y_base=lgb_basemodel.predict(x_test)
#Upsampling

y_upsampling=lgb_upsampling.predict(x_test)
#Downsampling

y_downsampling=lgb_downsampling.predict(x_test)
#Upweighting

y_upweighting=lgb_upweighting.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import scikitplot as skplt
#Basemodel

skplt.metrics.plot_confusion_matrix(y_test,y_base>0.5,normalize=False,figsize=(12,8),title='Confusion matrix for base model')  

plt.show()
#Upsampling

skplt.metrics.plot_confusion_matrix(y_test,y_upsampling>0.5,normalize=False,figsize=(12,8),title='Confusion matrix for upsampling model')  #0.5 is threshold value

plt.show()
#downsampling

skplt.metrics.plot_confusion_matrix(y_test,y_downsampling>0.5,normalize=False,figsize=(12,8),title='Confusion matrix for downsampling model')  #0.5 is threshold value

plt.show()
#Upweighting

skplt.metrics.plot_confusion_matrix(y_test,y_upweighting>0.5,normalize=False,figsize=(12,8),title='Confusion matrix for upweighting model')  #0.5 is threshold value

plt.show()
#Base model

cm_base=classification_report(y_test,y_base>0.5)

print(cm_base)
#Upsampling model

cm_up=classification_report(y_test,y_upsampling>0.5)

print(cm_up)
#Downsampling model

cm_up=classification_report(y_test,y_downsampling>0.5)

print(cm_up)
#Upweighting model

cm_upweight=classification_report(y_test,y_upweighting>0.5)  # 0.5 is threshold value

print(cm_upweight)