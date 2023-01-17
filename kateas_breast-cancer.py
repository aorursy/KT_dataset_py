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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
DataFrame = pd.read_csv("../input/data.csv")
DataFrame.info()
y = DataFrame.diagnosis.map({'M':1,'B':0})                          # M or B 
list = ['Unnamed: 32','id','diagnosis']
x = DataFrame.drop(list,axis = 1 )
x.head()
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=68)
clf = LogisticRegression()
clf.fit(X_train,y_train)                              
print(classification_report(y_test,clf.predict(X_test)))