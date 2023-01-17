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

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/weatherAUS.csv')
dataset.head()
dataset.isnull().sum()
dataset.shape
dataset.drop(labels = ['Date','Location','Evaporation','Sunshine','Cloud3pm','Cloud9am','RISK_MM'],axis = 1,inplace = True)
dataset.head()
dataset['RainToday'].replace({'No':0,'Yes':1},inplace = True)

dataset['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)

dataset.shape
dataset.dropna(inplace = True)
dataset.shape
categorical = ['WindGustDir','WindDir9am','WindDir3pm']
dataset = pd.get_dummies(dataset,columns = categorical,drop_first=True)
dataset.head()
dataset.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = dataset.drop(labels = ['RainTomorrow'],axis = 1)
x.shape
y = dataset['RainTomorrow']
x = sc.fit_transform(x)
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)
from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(n_estimators = 200,max_leaf_nodes = 1000)

rc.fit(x_train,y_train)
y_pred = rc.predict(x_test)
y_train_pred = rc.predict(x_train)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_train,y_train_pred))
confusion_matrix(y_train,y_train_pred)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print('Training accuracy ---->',accuracy_score(y_train,y_train_pred))

print('Testing accuracy  ---->',accuracy_score(y_test,y_pred))
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

y_train_pred = svc.predict(x_train)
print(classification_report(y_train,y_train_pred))
confusion_matrix(y_train,y_train_pred)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu',input_dim = 58))

classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))

classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))

classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
classifier.fit(x_train,y_train,epochs = 50,batch_size=10)
y_pred = classifier.predict_classes(x_test)

y_train_pred = classifier.predict_classes(x_train)
print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))