import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/diabetes.csv')
df.head(10)
x = df[[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]].values

y = df[['Outcome']].values
len(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_train))
print(len(x_test))
def metric(y,pred):
    avg = 'weighted'
    print('Precision score = ',metrics.precision_score(y, pred, average=avg))
    print('Recall score = ',metrics.recall_score(y, pred, average=avg))
    print('f1-score = ',metrics.f1_score(y, pred, average=avg))
t = time()
clf = DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train,y_train)
print('Train Time DTC = ',(time()-t))
pre = clf.predict(x_test)
print(pre)
print('Accuracy = ',clf.score(x_test,y_test))

metric(y_test,pre)
print('Time DTC = ',(time()-t))
t = time()
clf = SVC(kernel='linear',C=0.1)
print(clf)
clf.fit(x_train,y_train.ravel())
print('Train Time SVC = ',(time()-t))
pre = clf.predict(x_test)
print(pre)
print('Accuracy = ',clf.score(x_test,y_test))

metric(y_test,pre)
print('Time SVC = ',(time()-t))
t = time()
clf = KNeighborsClassifier(n_neighbors=12)
print(clf)
clf.fit(x_train,y_train)
print('Train Time KNC = ',(time()-t))
pre = clf.predict(x_test)
print(pre)
print('Accuracy = ',clf.score(x_test,y_test))

metric(y_test,pre)
print('Time KNC = ',(time()-t))
t = time()
clf = GaussianNB()
print(clf)
clf.fit(x_train,y_train)
print('Train Time GNB = ',(time()-t))
pre = clf.predict(x_test)
print(pre)
print('Accuracy = ',clf.score(x_test,y_test))

metric(y_test,pre)
print('Time GNB = ',(time()-t))
