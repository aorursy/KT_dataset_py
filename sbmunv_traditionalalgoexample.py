# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#codes are taken from different open source sites



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import Libraries 

import pandas as pd 

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm, tree

import xgboost

from sklearn.model_selection import train_test_split

#Read the data

data  =  pd.read_csv(r"../input/Iris.csv")

#Create Dependent and Independent Datasets based on our Dependent #and Independent features

X  = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm']]

y= data['Species']

#Split the Data into Training and Testing sets with test size as #30%

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True,random_state=1)
set(y_train)
classifiers = []



model1 = xgboost.XGBClassifier()

classifiers.append(model1)

model2 = svm.SVC()

classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()

classifiers.append(model3)

model4 = RandomForestClassifier()

classifiers.append(model4)
for clf in classifiers:

    clf.fit(X_train, y_train)

    y_pred= clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy of %s is %s"%(clf.__class__.__name__, acc))

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix of %s is \n %s"%(clf.__class__.__name__, cm))