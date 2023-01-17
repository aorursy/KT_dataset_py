# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data["banking_crisis"] = le.fit_transform(data["banking_crisis"])

data["banking_crisis"] = data["banking_crisis"].astype("category") 



data["cc3"] = le.fit_transform(data["cc3"])

data["cc3"] = data["cc3"].astype("category") 



data["country"] = le.fit_transform(data["country"])

data["country"] = data["country"].astype("category")



x = data.values[: , 0:13]

y = data["banking_crisis"]
print(x.shape)

print(data.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix



xtrain,xtest,ytrain,ytest = train_test_split(x, y,test_size=0.2)

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(xtrain,ytrain)

#Predict the response for test dataset

y_pred = clf.predict(xtest)

acc1 = metrics.accuracy_score(ytest, y_pred)

matrix=confusion_matrix(ytest,y_pred)
matrix
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier

clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(xtrain,ytrain)

y_pred2=clf.predict(xtest)

acc2 = metrics.accuracy_score(ytest, y_pred2)

matrix2=confusion_matrix(ytest,y_pred2)
acc2