# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pan

from sklearn.preprocessing import StandardScaler

import sklearn.preprocessing as pre

import keras as ks

import matplotlib.pyplot as plt

from keras.layers import Dense

from keras.callbacks import EarlyStopping

from keras.regularizers import l2

from keras.layers import Dropout

from keras import optimizers
#Given: Train and Test Data set - separately



#Reading Train Data set
#mine

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.iloc[0]

#mine
def checkfornull(dataset):

    return(dataset.columns[dataset.isnull().any()])
TrainData_X = train.drop(['ID_code', 'target'], axis=1)

TargetTrainData_Y = train['target']
#data_set_testing = readfile("test.csv")

TestData_X = test.drop(['ID_code'], axis=1) # excluding ID_code, 200 columns
# Check for null or NaN values

null_columns=checkfornull(train)

if(null_columns.size == 0):

    print("There are no columns with NULL values in the Training Data set.")

    

null_columns = checkfornull(test)

if(null_columns.size == 0):

    print("There are no columns with NULL values in the Testing Data set.")
train.target.value_counts()
f,ax=plt.subplots(1,2,figsize=(20,10))

train[train['target']==0].var_0.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('target= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train[train['target']==1].var_0.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('target= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
check_balance(train,'target')
training_features, test_features,training_target, test_target, = train_test_split(TrainData_X,TargetTrainData_Y,test_size = .1,random_state=12)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,test_size = .1,random_state=12)



sm = SMOTE(random_state=12, ratio = 1.0)

x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)

clf_rf.fit(x_train_res, y_train_res)
print('Validation Results') 

print(clf_rf.score(x_val, y_val)) 

print(recall_score(y_val, clf_rf.predict(x_val))) 

print('\nTest Results') 

print(clf_rf.score(test_features, test_target)) 

print(recall_score(test_target, clf_rf.predict(test_features))) 
# plot the accuracy and loss

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('Plot History: Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Plot History: Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predict = model.predict(scaled_test_data_X)



result = pan.DataFrame({"ID_code": data_set_testing['ID_code'], "target": predict[:,0]})

print(result.head())



result.to_csv("submission.Barbour.Mar152019.1.csv", index=False)