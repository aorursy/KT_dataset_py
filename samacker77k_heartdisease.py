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
df = pd.read_csv('../input/heart.csv')
df.describe()
df.shape
df.head()
features = df.iloc[:,:13].values
features[:5]
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()

scaler.fit_transform(features)
labels = df.iloc[:,13].values
labels[:5]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit_transform(features)
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.20,random_state=109)
from sklearn.svm import SVC
clf = SVC(C=0.5,kernel = 'linear')
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy = {0:.2f}".format(accuracy_score(preds,y_test)*100))

print("Precision:",round(precision_score(y_test, preds)*100,2))

print("Recall:",round(recall_score(y_test,preds)*100,2))