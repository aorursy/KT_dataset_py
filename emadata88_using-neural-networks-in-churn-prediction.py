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

import sklearn

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import keras

import theano

import tensorflow

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras.models import Sequential

from keras.layers import Dense, Dropout

from sklearn.metrics import confusion_matrix ,classification_report

from keras.wrappers.scikit_learn import KerasClassifier

# Read Data

cust = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Explore Dataset

print('Dimensions:{}'.format(cust.shape))

print(cust.dtypes)

cust.head()

cust.isnull().sum()
null_values=cust[cust['TotalCharges'] == ' ']

null_values
cust.drop(cust[(cust['TotalCharges'] == ' ')].index,inplace=True)

null_values=cust[cust['TotalCharges'] == ' ']

null_values

cust['TotalCharges']=cust['TotalCharges'].astype('float64')
# Remove Columns Customerid.



cust.drop(['customerID'],axis=1,inplace=True)





# Encode target feature as "Yes"=1 and "No"=0



cust['Churn'].replace({"Yes":1,"No":0},inplace=True)



#Encoding categorical data

d=cust.select_dtypes(include=['object'])

d=pd.get_dummies(d,prefix_sep='_',drop_first=True)

cust=cust.iloc[:,[1,4,17,18,19]]

cust=pd.concat([cust,d],axis=1)

cust['TotalCharges'].astype('float64')







# Splitting the dataset into Training and Test Set



X=cust.drop(['Churn'],axis=1)

y=cust['Churn']



X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=80)



print('Dimensions of the training feature table: {}'.format(X_train.shape))

print('Dimensions of the training target vector: {}'.format(y_train.shape))

print('Dimensions of the test feature table: {}'.format(X_test.shape))

print('Dimensions of the test target vector: {}'.format(y_test.shape))

#Feature Scaling

scal=StandardScaler()

X_train=scal.fit_transform(X_train)

X_test=scal.fit_transform(X_test)

# First Neural Network



def nn_classifier():

    nn = Sequential()

    nn.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=30)) # Initial Input and First hidden Layer

    nn.add(Dropout(p = 0.1)) #Dropout Reg

    nn.add(Dense(output_dim=16,init='uniform',activation='relu')) # Second hidden Layer

    nn.add(Dropout(p = 0.1)) #Dropout Reg

    nn.add(Dense(output_dim=1,init='uniform',activation='sigmoid')) # Output Layer

    nn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return nn

nn = KerasClassifier(build_fn= nn_classifier,batch_size= 10,nb_epoch=100)

acc = cross_val_score(estimator  = nn, X = X_train, y = y_train, cv = 10, n_jobs = -1)

print("Mean Accuracy : {}".format(acc.mean()))

print("Variance : {}".format(acc.std()))
def nn_classifier(optimizer):

    nn = Sequential()

    nn.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=30)) # Initial Input and First hidden Layer

    nn.add(Dropout(p = 0.1)) #Dropout Reg

    nn.add(Dense(output_dim=16,init='uniform',activation='relu')) # Second hidden Layer

    nn.add(Dropout(p = 0.1)) #Dropout Reg

    nn.add(Dense(output_dim=1,init='uniform',activation='sigmoid')) # Output Layer

    nn.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

    return nn

nn = KerasClassifier(build_fn= nn_classifier)

parameters = {'batch_size' : [20, 33 ], 

              'nb_epoch' : [100, 300],

              'optimizer': ['adam','rmsprop']}

gs = GridSearchCV(estimator = nn,

                 param_grid = parameters,

                 scoring = 'accuracy',

                 cv = 10)





gs = gs.fit( X_train, y_train)

opt_param = gs.best_params_

opt_acc = gs.best_score_

print("Optimal Parameters : {}".format(opt_param))

print("Optimal Accuracy : {}".format(opt_acc))