#libraries
import numpy as np 
import pandas as pd 
import os
import json
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')

train['DatasetType'] = 'train'
test['DatasetType'] = 'test'
all_data = pd.concat([train, test])
print(os.listdir("../input"))
train.drop('Description', axis=1).head()
train.info()
non_txt_features = train.columns.difference(["Name", "State", "RescuerID", "PetID", "DatasetType"])
train[non_txt_features].hist(figsize=(20,20))
non_txt_features = train.columns.difference(["Name", "State", "RescuerID", "PetID", "DatasetType", "AdoptionSpeed"])
test[non_txt_features].hist(figsize=(20,20))
all_data[non_txt_features].hist(figsize=(20,20))
import sklearn
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
l = LogisticRegression()
features = train.columns.difference(["Name", "Description", "PetID", "RescuerID", "AdoptionSpeed"])
train_x = train[features]
train_y = train["AdoptionSpeed"]
test_x = test[features]
from sklearn.model_selection import train_test_split
enc.fit(train_x)
x_train_1h = enc.transform(train_x)
train_xx,cv_x,train_yy,cv_y=train_test_split(x_train_1h,train_y,test_size=0.2)
l.fit(train_xx,list(train_yy))
y_pred = l.predict(cv_x)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(cv_y),y_pred)
accuracy
enc.fit(train_x)
X_train_one_hot = enc.transform(train_x)
X_test_one_hot = enc.transform(test_x)
l.fit(X_train_one_hot,train_y)
y_pred = l.predict(X_test_one_hot)
print(X_train_one_hot.shape)
y_pred
sub.head()
for i,val in enumerate(y_pred):
    sub.at[i,'AdoptionSpeed'] = val
sub.AdoptionSpeed = sub.AdoptionSpeed.astype(int)
sub.head()
sub.to_csv('submission.csv', index=False)
