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
diab=pd.read_csv("../input/diabetes.csv")
diab.head()
max(diab.Pregnancies) 
## Yok artık Lebron James.
diab[diab.Pregnancies>=15]
print(max(diab.BloodPressure))

print(min(diab.BloodPressure))
diab.BMI.describe()
diab[diab.BMI>50].BMI.describe()
diab[diab.BMI>50].Outcome
diab[diab.BMI>50]
diab[diab.BMI<=20]
x=diab.iloc[:,0:8]

y=diab.iloc[:,8:9]
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier() #Use Gini İndex

dt.fit(x_train,y_train)

ypred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
diab.tail()

selected=diab[["BMI","Age","Pregnancies","Glucose","Outcome"]]

x=selected.iloc[:,0:4]

y=selected.iloc[:,4:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier() #Use Gini İndex

dt.fit(x_train,y_train)

ypred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab[["BMI","Age","Pregnancies","Glucose","BloodPressure","Outcome"]]

x=selected.iloc[:,0:5]

y=selected.iloc[:,5:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier() #Use Gini İndex

dt.fit(x_train,y_train)

ypred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
x=diab.iloc[:,0:8]

y=diab.iloc[:,8:9]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier(max_depth=2) #Use Gini İndex

dt.fit(x_train,y_train)

ypred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))


from sklearn.tree import export_graphviz

export_graphviz(dt,out_file='tree_limited.dot',feature_names=list(x))

!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600

from IPython.display import Image

Image(filename = 'tree_limited.png')
x=diab.iloc[:,0:8]

y=diab.iloc[:,8:9]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier(max_depth=3) #Use Gini İndex

dt.fit(x_train,y_train)

ypred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))

export_graphviz(dt,out_file='tree_limited2.dot',feature_names=list(x))

!dot -Tpng tree_limited2.dot -o tree_limited2.png -Gdpi=600

from IPython.display import Image

Image(filename = 'tree_limited2.png')
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

ypred=svm.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab[["BMI","Age","Pregnancies","Glucose","BloodPressure","Outcome"]]

x=selected.iloc[:,0:5]

y=selected.iloc[:,5:]

svm=SVC()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

dt=DecisionTreeClassifier() #Use Gini İndex

svm.fit(x_train,y_train)

ypred=svm.predict(x_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab[["BMI","Age","Glucose","BloodPressure","Outcome"]]

x=selected.iloc[:,0:4]

y=selected.iloc[:,4:]

svm=SVC()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)

dt=DecisionTreeClassifier() #Use Gini İndex

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab[["BMI","Age","Pregnancies","Glucose","BloodPressure","Outcome"]]

x=selected.iloc[:,0:5]

y=selected.iloc[:,5:]

svm=SVC()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)

dt=DecisionTreeClassifier() #Use Gini İndex

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab

x=selected.iloc[:,0:8]

y=selected.iloc[:,8:]

svm=SVC()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)

dt=DecisionTreeClassifier() #Use Gini İndex

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))
selected=diab[["Age","Glucose","BMI","Outcome"]]

x=selected.iloc[:,0:3]

y=selected.iloc[:,3:]

svm=SVC()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(x_train)

X_test = sc_X.transform(x_test)

dt=DecisionTreeClassifier() #Use Gini İndex

svm.fit(X_train,y_train)

ypred=svm.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,ypred))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,ypred))