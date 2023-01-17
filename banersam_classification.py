# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')

data.head()
data['target'].value_counts()
feature_col =['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

#Input variables

X = data.loc[:,feature_col]

X= preprocessing.scale(np.array(X))

#Output variables

y = data['target']
# Splitting the data in train test

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.35)
# Model Selection Naive Bayes



from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB().fit(x_train,y_train)

predict = gnb.predict(x_test)



print(accuracy_score(predict,y_test))

print(confusion_matrix(predict,y_test))

print(classification_report(predict,y_test))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression().fit(x_train,y_train)

predict_lr = gnb.predict(x_test)



print(accuracy_score(predict_lr,y_test))

print(confusion_matrix(predict_lr,y_test))

print(classification_report(predict_lr,y_test))
from sklearn.neighbors import KNeighborsClassifier



knc = KNeighborsClassifier(n_neighbors=10).fit(x_train,y_train)

predict_knc = gnb.predict(x_test)



print(accuracy_score(predict_knc,y_test))

print(confusion_matrix(predict_knc,y_test))



from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=25).fit(x_train,y_train)

predict_rfc = rfc.predict(x_test)



print(accuracy_score(predict_rfc,y_test))

print(confusion_matrix(predict_rfc,y_test))
