# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read in train and test datasets.

train = pd.read_csv("../input/train.csv")
#Setup predictor and response variables

y_train=train['type']

index_train=train['id']

train=train.drop(['id','type'],axis=1)

train=pd.get_dummies(train)
#Setup train and validation cross validation datasets



from sklearn.cross_validation import train_test_split



X_train,X_validation,Y_train,Y_validation=train_test_split(train,y_train,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



#Logistic Regression without Standardization

lr=LogisticRegression()

lr.fit(X_train,Y_train)

y_validation_pred=lr.predict(X_validation)

print(accuracy_score(y_validation_pred,Y_validation))

#This gives an accuracy score of 0.6875
#Standardization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_std=sc.fit_transform(X_train)

X_validation_std=sc.transform(X_validation)
#Logistic Regression on standardized dataset

lr=LogisticRegression()

lr.fit(X_train_std,Y_train)

y_validation_pred=lr.predict(X_validation_std)

print(accuracy_score(y_validation_pred,Y_validation))

#This gives an accuracy score of 0.7767.....