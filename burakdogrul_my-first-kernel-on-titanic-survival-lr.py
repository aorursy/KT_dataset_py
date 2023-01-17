# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Data Load
data = pd.read_csv("../input/train.csv")
data.info()
data.head(10)
#Drop
data = data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"], axis=1)
data.dropna(axis=0, inplace=True)   
#Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data.info()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
x = data.iloc[:,1:].values
y = data.iloc[:,:1].values
g = sns.jointplot(data.Pclass, data.Survived, kind="kde", size=10)
plt.savefig('graph.png')
plt.show()
#Slicing Train and Test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=2)
#Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(solver="lbfgs",random_state=0)
logr.fit(X_train,y_train.ravel())

y_pred = logr.predict(X_test)
#Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Test accuracy {}".format(logr.score(X_test,y_test)))