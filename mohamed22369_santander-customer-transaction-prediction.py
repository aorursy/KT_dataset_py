# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

data.head()
data.head()
data.shape

data=data.drop('ID_code',axis=1)

dfset=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 

           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 

           ]
X,y=data.drop('target',axis=1),data['target']

X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50,shuffle=True)

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation, Dropout

import gc

classes=1

model = Sequential()

model.add(Dense(16,activation='relu', input_shape=(200,)))

model.add(Dense(64,activation='relu'))

model.add(Dense(classes,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,batch_size=2048,epochs=20,validation_data=(X_test,y_test))

sc_train = model.evaluate(X_train,y_train,verbose=0)

print('Accuracy',sc_train)
sc_test = model.evaluate(X_test,y_test,verbose=0)

print('Accuracy',sc_test)
pred0=model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,pred0.round()))
print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
classes=1

model1 = Sequential()

model1.add(Dense(16,activation='relu', input_shape=(200,)))

model1.add(Dense(64,activation='relu'))

model1.add(Dense(classes,activation='sigmoid'))

model1.summary()

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history2=model1.fit(X_train,y_train,batch_size=2048,epochs=20,validation_data=(X_test,y_test))
sc_train = model1.evaluate(X_train,y_train,verbose=0)

print('Accuracy',sc_train)

sc_train = model1.evaluate(X_test,y_test,verbose=0)

print('Accuracy',sc_train)

pred1=model1.predict(X_test)

print(classification_report(y_test,pred1.round()))
sam = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

dftest = dfset.loc[:,dfset.columns!='ID_code']

#dftest.head()

pred2 = model1.predict(dftest)

pred2

pred2.shape
s = pd.DataFrame({'ID_code':sam['ID_code'],'target':pred2.ravel()})

s.head()
s.to_csv('submit.csv',index=False)