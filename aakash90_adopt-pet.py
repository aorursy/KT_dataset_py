"""This is my first model. Please share your feedback/suggestion to help me improve this model. 

I am new to DS/ML to getting difficulties in my applying different trics."""
#Import libraries

import numpy as np 

import pandas as pd 

from sklearn import preprocessing

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.model_selection import train_test_split

import xgboost as xgb
#reading data

train= pd.read_csv('../input/pet-adopt/train.csv')  #Reading train and test files.



test= pd.read_csv('../input/pet-adopt/test.csv')



#Shape of our train and test data.



print("Train Shape: ",train.shape)

print("Test Shape: ", test.shape)
df=train[['length(m)','height(cm)']]

df['length(cm)'] = df['length(m)']*100
print(len(train[train['length(m)'] == 0]))

print(len(test[test['length(m)']==0]))

train['length(cm)'] = train['length(m)'].apply(lambda x: x*100) # convert length from cm to m. 

test['length(cm)'] = test['length(m)'].apply(lambda x: x*100)

train.drop('length(m)', axis=1, inplace=True)

test.drop('length(m)', axis=1, inplace=True)
# replace all 0 length with mean of lengths

val = train['length(cm)'].mean()

train['length(cm)'] = train['length(cm)'].replace(to_replace=0, value=val)

test['length(cm)'] = test['length(cm)'].replace(to_replace=0, value=val)
lbl = preprocessing.LabelEncoder()  #label encoding for categorical features.

train['color_type'] = lbl.fit_transform(train['color_type'])

test['color_type'] = lbl.fit_transform(test['color_type'])



train.drop(['issue_date', 'listing_date'], axis=1, inplace=True)

test.drop(['issue_date', 'listing_date'], axis=1, inplace=True)
X_train = train.drop(['pet_id','breed_category', 'pet_category'], axis=1)

y_train = train['breed_category']

X_test = test.drop('pet_id', axis=1).copy()
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.01).fit(X_train, y_train)

predictions = gbm.predict(X_test)

acc = round(gbm.score(X_train, y_train)*100, 2)

acc
X_train = train.drop(['pet_id','breed_category', 'pet_category'], axis=1)

y_train = train['pet_category']

X_test = test.drop('pet_id', axis=1).copy()
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.01).fit(X_train, y_train)

predictions = gbm.predict(X_test)

acc = round(gbm.score(X_train, y_train)*100, 2)

acc