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
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:27:45 2018

@author: AJAY
"""

import pandas as pd
dataframe=pd.read_csv("../input/train.csv")
dataframe2=pd.read_csv("../input/test.csv")
df=pd.DataFrame(dataframe,columns=['Sex','Age','Pclass'])
df3=pd.DataFrame(dataframe2,columns=['PassengerId'])
df0=pd.DataFrame(dataframe,columns=['Survived'])
df1=pd.DataFrame(dataframe2,columns=['Sex','Age','Pclass'])
df2=pd.DataFrame(dataframe2,columns=['Survived'])
X = df.iloc[:,0:-1].values
Y = df0.iloc[:, 0].values
X_test=df1.iloc[:,0:-1].values
Y_test=df3.iloc[:,0].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 1:2])
X_test[:, 1:2] = imputer.transform(X_test[:, 1:2])
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_test[:, 0] = labelencoder_X.fit_transform(X_test[:, 0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, Y)

y_pred = classifier.predict(X_test)

#np.savetxt("gender_submission.csv", zip(Y_test,y_pred),fmt='%10.0f', header="PassengerId,Survived",newline="\n",delimiter=',')


