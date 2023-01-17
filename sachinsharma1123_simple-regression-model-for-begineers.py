# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_test=pd.read_csv('/kaggle/input/airbnb-listings-in-major-us-cities-deloitte-ml/test.csv')
df_test
train_df=pd.read_csv('/kaggle/input/airbnb-listings-in-major-us-cities-deloitte-ml/train.csv')
train_df
#drop unnecessary features from both train and test set



train_df=train_df.drop(['first_review','host_response_rate','review_scores_rating','neighbourhood','last_review','thumbnail_url','zipcode'],axis=1)
df_test=df_test.drop(['first_review','host_response_rate','review_scores_rating','neighbourhood','last_review','thumbnail_url','zipcode'],axis=1)
Id=df_test['id']
train_df.corr()
train_df=train_df.drop(['amenities','longitude','latitude','host_has_profile_pic','host_identity_verified','host_since','name','id'],axis=1)
df_test=df_test.drop(['amenities','longitude','latitude','host_has_profile_pic','host_identity_verified','host_since','name','id'],axis=1)
train_df=train_df.drop(['city','description'],axis=1)
df_test=df_test.drop(['city','description'],axis=1)
train_df
train_df.isnull().sum()
df_test.isnull().sum()
#for train data

dict_1={}



dict_1=dict(train_df.isnull().sum())
#for test_data

dict_2={}

dict_2=dict(df_test.isnull().sum())
#aggregating all the nan values form both datasets form imputation

list_1=[]

list_2=[]

for key,values in dict_1.items():

    if train_df[key].dtype=='object' and values>0:

        list_1.append(key)

    if train_df[key].dtype!='object' and values>0:

        list_2.append(key)
list_1,list_2
list_3=[]

list_4=[]

for key,values in dict_2.items():

    if df_test[key].dtype=='object' and values>0:

        list_3.append(key)

    if df_test[key].dtype!='object' and values>0:

        list_4.append(key)
list_2
for i in list_2:

    train_df[i]=train_df[i].fillna(train_df[i].mode()[0])
for i in list_2:

    train_df[i]=train_df[i].astype('int64')
list_4
for i in list_4:

    df_test[i]=df_test[i].fillna(df_test[i].mode()[0])
for i in list_4:

    df_test[i]=df_test[i].astype('int64')
train_df.isnull().sum()
df_test.isnull().sum()
y=train_df['log_price']
train_df=train_df.drop(['log_price'],axis=1)
#let separate the catrgorical features for encoding process

list_cate=[]

for i in list(train_df.columns):

    if train_df[i].dtype=='object':

        list_cate.append(i)

   
test_cate=[]

for i in list(df_test.columns):

    if df_test[i].dtype=='object':

        test_cate.append(i)

    
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for i in list_cate:

    

    train_df[i]=le.fit_transform(train_df[i])
for i in test_cate:

    df_test[i]=le.fit_transform(df_test[i])
df_test
train_df['cleaning_fee']=le.fit_transform(train_df['cleaning_fee'])
df_test['cleaning_fee']=le.fit_transform(df_test['cleaning_fee'])
x=train_df
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
from sklearn.metrics import r2_score

score_1=r2_score(y_test,pred_1)
score_1
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

rgb=GradientBoostingRegressor()

param_grid={'n_estimators':[100,500], 

            'learning_rate': [0.1,0.05,0.02],

            'max_depth':[4], 

            'min_samples_leaf':[3], 

            'max_features':[1.0] }

    
clf=GridSearchCV(rgb,param_grid,cv=5,verbose=0)
clf.fit(x_train,y_train)
rgb=GradientBoostingRegressor(learning_rate=0.1,max_depth=4,max_features=1.0,n_estimators=500)
rgb.fit(x_train,y_train)

pred_2=rgb.predict(x_test)
score_2=r2_score(y_test,pred_2)
score_2
preds=rgb.predict(df_test)
submission=pd.DataFrame({'id':Id,

                       'predictions':preds})
submission