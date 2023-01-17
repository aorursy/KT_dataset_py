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
import pandas as pd
import numpy as np

data = pd.read_csv("../input/train.csv")

g=data.drop("Cabin",axis=1,inplace=True)
k=data.dropna(inplace=True)
l=pd.get_dummies(data["Sex"],drop_first=True)
m=pd.get_dummies(data["Embarked"],drop_first=True)
n=pd.get_dummies(data["Pclass"],drop_first=True)
o= pd.concat([data,l,m,n],axis=1)
p=o.drop(["Sex","Name","Pclass","Embarked","Ticket"],axis=1,inplace=True)


X_train= o.drop("Survived",axis=1)
y_train= o["Survived"]


data1 = pd.read_csv("../input/test.csv")

g1=data1.drop("Cabin",axis=1,inplace=True)
k1=data1.dropna(inplace=True)
l1=pd.get_dummies(data1["Sex"],drop_first=True)
m1=pd.get_dummies(data1["Embarked"],drop_first=True)
n1=pd.get_dummies(data1["Pclass"],drop_first=True)
o1= pd.concat([data1,l1,m1,n1],axis=1)
p1=o1.drop(["Sex","Name","Pclass","Embarked","Ticket"],axis=1,inplace=True)


X1 = o1
X_test = X1

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

accuracy_log = logreg.score(X_train, y_train)
print("The accuracy for this model with Logistic Regression is: " + str(int(accuracy_log*100)), "Percent")

