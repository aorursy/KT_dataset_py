import numpy as np

import pandas as pd

import os

import string
dataset = pd.read_csv("../input/plant-village/plant_village.csv")
dataset.head(30)
type(dataset)
dataset1 = pd.read_csv("../input/leafimages/images.csv")
img_files=dataset1['0'].tolist()
img_files[21915]
breakpoints = [1000,1269,1270,1549,1550,1673,1674,2386,2387,2972,2973,3482,3483,3840,3841,4031,4032,4550,4551,4924,4925,5369,5370,5845,5846,6397,6398,6817,6818,6988,6989,9309,9310,10271,10272,10379,10380,10807,10808,11384,11385,11796,11797,12176,12177,12240,12241,12480,12481,14397,14398,15146,15147,15536,15537,15750,15751,16542,16543,16947,16948,17673,17674,18033,18034,18767,18768,19487,19488,20034,20035,22134,22135,22273,22274,22916]

print(len(breakpoints))


target_list = []

for file in img_files:

    target_num = int(file.split(".")[0])

    #print(target_num)

    flag = 0

    i = 0 

    for i in range(0,len(breakpoints),2):

        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):

            flag = 1

            break

    

    if(flag==1):

        target = int((i/2))

        target_list.append(target)
y = np.array(target_list)

y


X = dataset.iloc[:,1:]
#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)
X_train.head(5)
X_train.shape
y_train[0:5]
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
X_train[0:2]
y_train[0:2]
#Applying SVM classifier model
from sklearn import svm
clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['rbf'],

               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],

               'C': [1, 10, 100, 1000]},

              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

             ]
parameters1 = [{'kernel': ['rbf'],

               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5,0.6,0.7,0.8,1.0,2.2],

               'C': [1, 2,4,8,16,32,64,128,170,216,300,414,888,999, 1000]},

              {'kernel': ['linear'], 'C': [1, 2,4,8,16,32,64,128,216,414,888,999, 1000]}

             ]
svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)

svm_clf.fit(X_train, y_train)
svm_clf.best_params_
clf = svm.SVC(C=200.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=2.1, kernel='poly',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred)