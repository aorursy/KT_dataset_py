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
heart = pd.read_csv("../input/heart.csv")

heart.head(10)
heart.isnull().sum()

#Wow this data looks so Clean, Let's See what else can we do!
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix

from sklearn.model_selection import train_test_split



X = heart.loc[:,'thal':].as_matrix().astype('float')

y= heart['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape)

print(y_train.shape,y_test.shape)
model = LogisticRegression(random_state=0)

model.fit(X_train,y_train)

print('Score: {0:.2f}'.format(model.score(X_test,y_test)))

print('Confusion Matrix : \n{0}'.format(confusion_matrix(y_test,model.predict(X_test))))

print("Accuracy of Model : {0:.2f}".format(accuracy_score(y_test,model.predict(X_test))))

print('Precision: {0:.2f}'.format(precision_score(y_test,model.predict(X_test))))

print('Recall: {0:.2f}'.format(recall_score(y_test,model.predict(X_test))))

# Wow Our One Time Model gives 100% Accuracy. That is some serious Model.
#To chaeck, let's pick a Sklearn Dummy Model

from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy = 'most_frequent',random_state = 0)

dummy.fit(X_train,y_train)
print('Accuracy: {0:.2f}'.format(accuracy_score(y_test,dummy.predict(X_test))))

print('Confusion Matrix : \n{0}'.format(confusion_matrix(y_test,dummy.predict(X_test))))

print('Precision: {0:.2f}'.format(precision_score(y_test,dummy.predict(X_test))))

print('Recall: {0:.2f}'.format(recall_score(y_test,dummy.predict(X_test))))

# So to compare, we can See that a most Requent Used Dummy Model has 56% of Accuracy.!