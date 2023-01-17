# importing Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as  sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)
# Reading the Data

train=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv',index_col='pet_id',parse_dates=['issue_date','listing_date'])

test=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv',index_col='pet_id',parse_dates=['issue_date','listing_date'])
#Displaying first five rows of train Dataset

train.head()
#shape of train and test dataset

train_obs,train_ftr=train.shape

test_obs,test_ftr=test.shape

print("Train Dataset has {} observations and Test Dataset has {} Dataset".format(train_obs,test_obs))
# column datatype 

train.info()
# Summary Statistics

train.describe()
# Null Values in Dataset

train.isna().sum()
# exploring the NAN values in 'condition' column

train_category_na_df=train[train['condition'].isna()]

train_category_na_df.groupby(['pet_category'])['breed_category'].value_counts()
# groupby to find minimum height

train.groupby(['pet_category','breed_category'])['height(cm)'].min()
# groupby to find maximum height

train.groupby(['pet_category','breed_category'])['height(cm)'].max()
train.groupby(['pet_category'])['breed_category'].value_counts(sort=False)
train_length_na_df=train[train['length(m)']==0]

train_length_na_df.groupby(['pet_category'])['breed_category'].value_counts()
# removing observation with length 0

train=train[train['length(m)']!=0]

train.shape
# exploring whether height has 0 cm

train[train['height(cm)']==0].shape
# X1 column - value counts

train.X1.value_counts(sort=False)
# X2 column - value counts

train.X2.value_counts(sort=False)
# train.condition.value_counts()

# train['length(m)'].value_counts()

## Pet Category- Category Distribution

# train.pet_category.value_counts(sort=False)

# train.breed_category.value_counts()
train.groupby(['pet_category'])['color_type'].value_counts(sort=False,normalize=True)*100
# cocatenating train and test set

train['train_or_test']='train'

test['train_or_test']='test'

dataframe=pd.concat([train,test],axis=0)



#length(cm)= length(m)*100

dataframe['length(cm)']=dataframe['length(m)']*100



#area(cm^2)= length(cm)*height(cm)

dataframe['area(cm^2)']=dataframe['length(cm)']*dataframe['height(cm)']
# fill NAN value with 3

dataframe['condition'].fillna(3,inplace=True)
dataframe['time_for_listing']=(dataframe['listing_date']-dataframe['issue_date'])/1000000000000

dataframe['time_for_listing']=pd.to_numeric(dataframe['time_for_listing'])
dataframe['Year_arrival'] = (dataframe['listing_date']).dt.year

dataframe['Month_arrival'] = (dataframe['listing_date']).dt.month

dataframe['Day_arrival'] = (dataframe['listing_date']).dt.day

dataframe['Dayofweek_arrival'] = (dataframe['listing_date']).dt.dayofweek

dataframe['DayOfyear_arrival'] = (dataframe['listing_date']).dt.dayofyear

dataframe['Week_arrival'] = (dataframe['listing_date']).dt.week

dataframe['Quarter_arrival'] = (dataframe['listing_date']).dt.quarter 







dataframe['Year_issue'] = (dataframe['issue_date']).dt.year

dataframe['Month_issue'] = (dataframe['issue_date']).dt.month

dataframe['Day_issue'] = (dataframe['issue_date']).dt.day

dataframe['Dayofweek_issue'] = (dataframe['issue_date']).dt.dayofweek

dataframe['DayOfyear_issue'] = (dataframe['issue_date']).dt.dayofyear

dataframe['Week_issue'] = (dataframe['issue_date']).dt.week

dataframe['Quarter_issue'] = (dataframe['issue_date']).dt.quarter 







dataframe['year_took']=dataframe['Year_arrival']-dataframe['Year_issue']

dataframe['months_took']=dataframe['Month_arrival']-dataframe['Month_issue']

dataframe['days_took']=dataframe['Day_arrival']-dataframe['Day_issue']
train=dataframe[dataframe['train_or_test']=='train']

test=dataframe[dataframe['train_or_test']=='test']



train_X=train.drop(['issue_date','color_type','listing_date','length(m)','breed_category', 'pet_category','train_or_test'],axis=1)

train_y=train['pet_category']

final_test_X=test.drop(['issue_date','color_type','listing_date','length(m)','breed_category', 'pet_category','train_or_test'],axis=1)
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import xgboost as XGB



X_train,X_test,y_train,y_test=train_test_split(train_X,train_y,test_size=0.2,stratify=train_y)
xg_cl=XGB.XGBClassifier()



xg_cl.fit(X_train,y_train)

y_train_predict=xg_cl.predict(X_train)

y_predict=xg_cl.predict(X_test)

test['pet_category']=xg_cl.predict(final_test_X)

print(accuracy_score(y_train,y_train_predict))

print(accuracy_score(y_test,y_predict))

print(confusion_matrix(y_test,y_predict))

print(f1_score(y_test,y_predict,average='weighted'))
XGB.plot_importance(xg_cl)
train_X=train.drop(['issue_date','color_type','listing_date','length(m)','breed_category','train_or_test'],axis=1)

train_y=train['breed_category']

final_test_X=test.drop(['issue_date','color_type','listing_date','length(m)', 'breed_category','train_or_test'],axis=1)





xg_cl=XGB.XGBClassifier()



xg_cl.fit(X_train,y_train)

y_train_predict=xg_cl.predict(X_train)

y_predict=xg_cl.predict(X_test)

test['breed_category']=xg_cl.predict(final_test_X)

print(accuracy_score(y_train,y_train_predict))

print(accuracy_score(y_test,y_predict))

print(confusion_matrix(y_test,y_predict))

print(f1_score(y_test,y_predict,average='weighted'))
XGB.plot_importance(xg_cl)
dfy=test[['breed_category','pet_category']]

dfy.head(1000)
dfy.to_csv('output.csv')