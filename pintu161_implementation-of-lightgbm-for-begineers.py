import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
#Importing the dataset
dataset = pd.read_csv('../input/Social_Network_Ads.csv')
X= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
dataset.head()
#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


import lightgbm as lgb

d_train = lgb.Dataset(x_train, label= y_train)
params = {}
params['learning_rate']= 0.003
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='binary_logloss'
params['sub_feature']=0.5
params['num_leaves']= 10
params['min_data']=50
params['max_depth']=10
clf= lgb.train(params, d_train, 100)
y_pred = clf.predict(x_test)

#convert into binary values

for i in range(0,100):
    if (y_pred[i] >= 0.5):
        y_pred[i] = 1
    else:
        y_pred[i] =0
len(y_pred)        
            

#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
accuracy
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm