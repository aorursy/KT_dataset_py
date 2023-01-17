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
df = pd.read_csv("../input/sensor_readings_24.csv")

df = df.replace('Move-Forward',0)
df = df.replace('Slight-Right-Turn',1)
df = df.replace('Sharp-Right-Turn',2)
df = df.replace('Slight-Left-Turn',3)

data = pd.DataFrame.as_matrix(df)
s = int(0.7*data.shape[0])

X_train = data[0:s,0:24]
X_test = data[s:data.shape[0],0:24]

y = data[0:s,24]

y_train = y.astype(int)
y_test = data[s:data.shape[0],24].astype(int)


from sklearn import linear_model
model = linear_model.LogisticRegression(penalty ='l2',max_iter=500,multi_class= 'ovr')

model.fit(X_train, y_train)



y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

from sklearn import metrics
acc1 = metrics.accuracy_score(y_test,y_pred_test)
acc = metrics.accuracy_score(y_train,y_pred_train)

print ("Accuracy for train set:"),
print (acc)
print ("Accuracy for test set:"),
print (acc1)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X_train, y_train)

y_pred_test = classif.predict(X_test)
y_pred_train = classif.predict(X_train)

from sklearn import metrics
acc1 = metrics.accuracy_score(y_test,y_pred_test)
acc = metrics.accuracy_score(y_train,y_pred_train)

print ("Accuracy for train set:"),
print (acc)
print ("Accuracy for test set:"),
print (acc1)