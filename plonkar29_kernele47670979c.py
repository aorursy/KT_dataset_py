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
df= pd.read_csv("../input/diabetes.csv")
df.head()
features = df[["Pregnancies","Glucose","Insulin","DiabetesPedigreeFunction"]]
labels = df.Outcome
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
features_train,features_test,labels_train,labels_test= train_test_split(features,labels,test_size=0.4)
clf= GaussianNB()
clf.fit(features_train,labels_train)
clf.score(features_test,labels_test)
from sklearn.metrics import confusion_matrix
var= clf.predict(features_test)
var