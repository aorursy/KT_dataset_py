# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import warnings
import seaborn as sns



train=pd.read_json('../input/train.json')
test=pd.read_json('../input/test.json')
train.head()

train['description'].isnull().sum()

train.shape
level=train['interest_level'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(level.index,level.values,alpha=0.8)
plt.ylabel("no of occurences")
plt.xlabel("Interest level")
plt.show()
counting=train['bathrooms'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(counting.index,counting.values,alpha=0.8)
plt.ylabel('Frequency')
plt.xlabel('bathrooms')
plt.show()
counting=train['bedrooms'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(counting.index,counting.values,alpha=0.8)
plt.ylabel('Frequency')
plt.xlabel('bathrooms')
plt.show()
train['num_photos']=train['photos'].apply(len)
train['len_description']=train['description'].apply(len)
train['num_features']=train['features'].apply(len)
train['created']=pd.to_datetime(train['created'])
train['year']=train['created'].dt.year
train['month']=train['created'].dt.month
train['day']=train['created'].dt.day

train.head()
y=train['interest_level']

y
train=train.drop(['building_id','created','description','display_address','interest_level','features','manager_id','photos','street_address'],axis=1)
train

# training the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X_train,X_val,y_train,y_val=train_test_split(train,y,test_size=0.33)
# random Forest Classifier

clf=RandomForestClassifier(n_estimators=1000)
clf.fit(X_train,y_train)
y_val_predicted=clf.predict_proba(X_val)
#print(accuracy(y_val,y_val_predicted,noramlise=True,sample_weight=None))
log_loss(y_val,y_val_predicted)
# df=pd.read_json(open("..input/test.json","r"))
# print(df.shape)
# df['num_photos']=df['photos'].apply(len)
# df['len_description']=df['description'].apply(len)
# df['num_features']=df['features'].apply(len)
# df['created']=pd.to_datetime(df['created'])
# df['year']=df['created'].dt.year
# df['month']=df['created'].dt.month
# df['day']=df['created'].dt.day

# df=df.drop(['building_id','created','description','display_address','interest_level','features','manager_id','photos','street_address'],axis=1)


# X=df
# y=clf.predict_proba(X)

