# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/email-spam-classification-dataset-csv/emails.csv")

df.head(20)
df.isnull().sum()
df.describe()
df.corr()
X = df.iloc[:,1:3001]

X
Y = df.iloc[:,-1].values

Y
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.25)
mnb = MultinomialNB(alpha=1.9)         # alpha by default is 1. alpha must always be > 0. 

# alpha is the '1' in the formula for Laplace Smoothing (P(words))

mnb.fit(train_x,train_y)

y_pred1 = mnb.predict(test_x)

print("Accuracy Score for Naive Bayes : ", accuracy_score(y_pred1,test_y))
svc = SVC(C=1.0,kernel='rbf',gamma='auto')         

# C here is the regularization parameter. Here, L2 penalty is used(default). It is the inverse of the strength of regularization.

# As C increases, model overfits.

# Kernel here is the radial basis function kernel.

# gamma (only used for rbf kernel) : As gamma increases, model overfits.

svc.fit(train_x,train_y)

y_pred2 = svc.predict(test_x)

print("Accuracy Score for SVC : ", accuracy_score(y_pred2,test_y))
rfc = RandomForestClassifier(n_estimators=100,criterion='gini')

# n_estimators = No. of trees in the forest

# criterion = basis of making the decision tree split, either on gini impurity('gini'), or on infromation gain('entropy')

rfc.fit(train_x,train_y)

y_pred3 = rfc.predict(test_x)

print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,test_y))