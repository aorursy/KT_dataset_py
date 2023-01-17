import numpy as np

import pandas as pd



import os

import string

from sklearn.externals import joblib
import cv2

import numpy as np

import pandas as pd

import string

#import mahotas as mt

from sklearn.externals import joblib

from matplotlib import pyplot as plt

%matplotlib inline
dataset = pd.read_csv("../input/original/original.csv")
dataset.head(15)
images = pd.read_csv("../input/images/images.csv")

img_files=images['0'].tolist()

img_files[1000]
breakpoints = [1000,1629,1630,2250,2251,2525,2526,4170,4171,5672,5673,6526,6527,7578,7579,8091,8092,9283,9284,10445,10446,11430,11431,12610,12611,13994,13995,14417,14418,15493,15494,21000,21001,23297,23298,23657,23658,24654,24655,26132,26133,27132,27133,27284,27285,28284,28285,28655,28656,33745,33746,35580,35581,36036,36037,37145,37146,39272,39273,40272,40273,41863,41864,43772,43773,44724,44725,46495,46496,48171,48172,49575,49576,49948,49949,55305]

print(len(breakpoints))


target_list = []

for file in img_files:

    

    target_num = int(file.split(".")[0])

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

X.head(5)
#Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)

X_train.head(5)
X_train.shape
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

X_train[0:2]
from sklearn import svm





clf = svm.SVC(C=160.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
filename = 'finalized_model.sav'

joblib.dump(clf, filename)


loaded_model = joblib.load(filename)

y_pred = loaded_model.predict(X_test)

metrics.accuracy_score(y_test, y_pred)