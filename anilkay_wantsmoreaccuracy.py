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
data=pd.read_csv("../input/datav2.csv")
data.head()
cins=data["Cinsiyet"]

cins[cins=="Erkek"]=1

cins[cins=="Kadın"]=0

data["Cinsiyet"]=cins
data.head()
set(data["Yas"])
data.shape
from sklearn.preprocessing import LabelEncoder

lr=LabelEncoder()

data.replace(('Evet', 'Hayır'), (1, 0), inplace=True)

data.head()
lr=LabelEncoder()

data["Bolge"]=lr.fit_transform(data["Bolge"])
data.head()
lr=LabelEncoder()

data["Egitim"]=lr.fit_transform(data["Egitim"])

data.head()
lr=LabelEncoder()

data["parti"]=lr.fit_transform(data["parti"])

data.head()
lr=LabelEncoder()

data["Yas"]=lr.fit_transform(data["Yas"])

data.head()
x=data.iloc[:,1:15].values

y=data.iloc[:,15:16].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
from sklearn.svm import SVC

svm=SVC()

svm.fit(X_train,y_train)

ypr=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypr))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypr))
from sklearn.manifold import TSNE

xem=TSNE(n_components=2,n_iter=1000).fit(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=1999)

svm=SVC()

svm.fit(X_train,y_train)

ypr=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypr))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypr))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

knn.fit(X_train,y_train)

ypr=knn.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypr))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypr))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(X_train,y_train)

ypr=rfc.predict(X_test)

print(confusion_matrix(y_test,ypr))

print(accuracy_score(y_test,ypr))
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(X_train,y_train)

ypr=dtc.predict(X_test)

print(confusion_matrix(y_test,ypr))

print(accuracy_score(y_test,ypr))
from sklearn.svm import SVC

svm=SVC()

svm.fit(X_train,y_train)

ypr=svm.predict(X_test)

print(confusion_matrix(y_test,ypr))

print(accuracy_score(y_test,ypr))


from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



scores = ['precision', 'accuracy']

clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring="accuracy")

clf.fit(X_train, y_train)

print(clf.best_params_)
from sklearn.svm import SVC

svm=SVC(C=10,gamma=0.001,kernel="rbf")

svm.fit(X_train,y_train)

ypr=svm.predict(X_test)

print(confusion_matrix(y_test,ypr))

print(accuracy_score(y_test,ypr))