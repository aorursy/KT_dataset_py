import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import keras
from keras.layers import *
%matplotlib inline
df = pd.read_csv("../input/train.csv")
testset = pd.read_csv("../input/test.csv")
df.head()
testset.head()
ids = testset['Id']
df = df.drop('Id',axis = 1)
testset = testset.drop('Id',axis = 1)
df.info()
df.isnull().sum()
testset.isnull().sum()
df.sum()
testset.sum()
df = df.drop(['Soil_Type7', 'Soil_Type15'],axis =1)
testset = testset.drop(['Soil_Type7', 'Soil_Type15'],axis =1)

df[df['Soil_Type8'] == 1]
df[df['Soil_Type25'] == 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
labels = df.Cover_Type
#labels = pd.get_dummies(labels)
labels = labels.values
features = df.drop('Cover_Type',axis =1)
features.shape
testset.shape # since we dont have the covertype already.
features = features.values
testset = testset.values
features[0]
testset[0]
features = scaler.fit_transform(features)
testset = scaler.transform(testset)
features[0]
testset[0]
print(type(labels))
print(type(features))
print(type(testset))

labels = labels - 1
train_x,test_x,train_y,test_y = train_test_split(features,labels)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
train_y
import lightgbm as lgb
from sklearn.metrics import accuracy_score
gbm = lgb.LGBMClassifier(objective="mutliclass",n_estimators=10000)
gbm.fit(train_x,train_y,early_stopping_rounds = 100, eval_set = [(test_x,test_y)],verbose = 300)
ypred1 = gbm.predict(test_x)
ypred1
accuracy_score(test_y,ypred1)
labels
gbm1 = lgb.LGBMClassifier(objective="mutliclass",n_estimators=4000)
gbm1.fit(features,labels,verbose = 1000)
finalval = gbm1.predict(testset)
covertype = finalval + 1
sub = pd.DataFrame({'Id':ids,'Cover_Type':covertype})
output = sub[['Id','Cover_Type']]
output.to_csv("output1.csv",index = False)