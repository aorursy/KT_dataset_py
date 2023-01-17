# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#libraries

import numpy as np 

import pandas as pd 

import os

import json

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report

import lightgbm as lgb

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import itertools

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')
# features to use: 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',

#       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

#       'Sterilized', 'Health', 'Quantity', 'Fee', 'State',

#       'VideoAmt', 'PhotoAmt'

feat_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 22]

lab_idx = 23
x_train = train.iloc[:, feat_idx].values

x_test = test.iloc[:, feat_idx].values

y_train = train.iloc[:, lab_idx].values

print("Training:", x_train.shape)

print("Test:", x_test.shape)
def init_lgbm(x, y, lr, sub_f, n_l, max_depth):

    d_train = lgb.Dataset(x, label=y)

    params = {}

    params['learning_rate'] = lr

    params['boosting_type'] = 'gbdt'

    params['objective'] = 'multiclass'

    params['num_class']= 5

    params['metric'] = 'multi_logloss'

    params['sub_feature'] = sub_f

    params['num_leaves'] = n_l

    params['min_data'] = 100

    params['max_depth'] = max_depth

    clf = lgb.train(params, d_train, 100)

    return clf
clf = init_lgbm(x_train, y_train, 0.01331578947368421, 0.8, 50, 10)
#Prediction

y_lgbm_pred=clf.predict(x_test)
print(y_lgbm_pred.shape)
data_tr = pd.read_csv("../input/train/train.csv")
Data_Tr = data_tr.drop(['Name', 'RescuerID','Description','PetID'], axis=1)
R_Data_Tr = np.asmatrix(Data_Tr)
X_Data_Tr = R_Data_Tr[:,0:19]

Y_Data_Tr = R_Data_Tr[:,19]
#Encode class values as integers

Y_Data_Tr = np.ravel(y_train)

encoder_Tr = LabelEncoder()

encoder_Tr.fit(Y_Data_Tr)

encoded_Y_Tr = encoder_Tr.transform(Y_Data_Tr)



# convert integers to dummy variables (i.e. one hot encoded)

dummy_y_Tr = np_utils.to_categorical(encoded_Y_Tr)
####### Feature Scaling



sc = StandardScaler()

X_train = sc.fit_transform(x_train)
# Define the model



#Initialising the ANN

model = Sequential()

#Adding the input layer and the first hidden layet

model.add(Dense(units = 10, kernel_initializer = 'uniform', input_dim = 19, activation = 'sigmoid'))

#Adding the second hidden layer

model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'tanh'))



model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'tanh'))



model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'tanh'))







model.add(Dense(units = 5, kernel_initializer = 'uniform', activation='softmax'))



#Compile model

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(X_train.shape)
model.fit(X_train, dummy_y_Tr, batch_size = 10, epochs = 100)
data_Te = pd.read_csv("../input/test/test.csv")
Data_Te = data_Te.drop(['Name', 'RescuerID','Description','PetID'], axis=1)
R_Data_Te = np.asmatrix(Data_Te)
X_Data_Te = R_Data_Te[:,0:19]
sc = StandardScaler()

X_test = sc.fit_transform(X_Data_Te)
y_MLP_pred_Te = model.predict(X_test)
print(y_MLP_pred_Te.shape)
y_pred_tot = np.concatenate((y_lgbm_pred, y_MLP_pred_Te), axis=1)



y_pred = np.argmax(y_pred_tot, axis=1)



for i in range(len(y_pred)):

    if (y_pred[i] > 4):

        y_pred[i] = y_pred[i] - 5

submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': y_pred.astype(np.int32)})

print(submission.head())

submission.to_csv('submission.csv', index=False)