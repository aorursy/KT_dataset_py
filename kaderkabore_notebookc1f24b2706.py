# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = df.drop("Unnamed: 32",1)

df.diagnosis.unique()

df.head()



# Any results you write to the current directory are saved as output.
from sklearn import tree

from sklearn import svm

from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

import pandas as pd



from sklearn.naive_bayes import GaussianNB



df = pd.read_csv("../input/data.csv",header = 0)

df.head()



from sklearn import tree

from sklearn import svm

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn.linear_model import LogisticRegression







from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



from sklearn.neural_network import MLPClassifier





df = pd.read_csv("../input/data.csv",header = 0)

df.head()



df = df.drop("id",1)

df = df.drop("Unnamed: 32",1)

df.diagnosis.unique()

df.head()





d = {'M' : 0, 'B' : 1}

df['diagnosis'] = df['diagnosis'].map(d)

features = list(df.columns[1:31])

print(features)





x = df[features]

y = df["diagnosis"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)



print('justtt xxxx')







Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(X_train,y_train)





accuracy = Tree.score(X_test, y_test)

print(accuracy)



Kfold = KFold(len(df),n_folds = 10,shuffle = False)

print("KfoldCrossVal score using Decision Tree is %s" %cross_val_score(Tree,x,y,cv=10).mean())







Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)





Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())





svc = svm.SVC(kernel='linear',C=1).fit(x,y)

Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc,x,y,cv=10).mean())





dt = Tree.fit(X_train,y_train)

y_pred = dt.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy =  dt.score(X_test, y_test)

print('tree')

print(accuracy)







rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy =  rf.score(X_test, y_test)

print('forest')

print(accuracy)





sm = svc.fit(X_train,y_train)

y_pred = sm.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy = sm.score(X_test, y_test)

print('svm')

print(accuracy)



clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print('neighbors')

print(accuracy)



model = GaussianNB()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print('naives')

print(accuracy)







model2 = MLPClassifier()

model2.fit(X_train, y_train)

accuracy2 = model2.score(X_test, y_test)

print('neural')

print(accuracy2)







model3 = LogisticRegression()

model3.fit(X_train, y_train)

accuracy2 = model3.score(X_test, y_test)

print('Logistic')

print(accuracy2)





model4 = ExtraTreesClassifier()

model4.fit(X_train, y_train)

accuracy4 = model4.score(X_test, y_test)

print('extraa')

print(accuracy4)





from sklearn import tree

from sklearn import svm

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn.linear_model import LogisticRegression







from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



from sklearn.neural_network import MLPClassifier





df = pd.read_csv("../input/data.csv",header = 0)

df.head()



df = df.drop("id",1)

df = df.drop("Unnamed: 32",1)

df.diagnosis.unique()

df.head()





d = {'M' : 0, 'B' : 1}

df['diagnosis'] = df['diagnosis'].map(d)

features = list(df.columns[1:31])

print(features)

from sklearn import tree

from sklearn import svm

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn.linear_model import LogisticRegression







from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



from sklearn.neural_network import MLPClassifier





df = pd.read_csv("../input/data.csv",header = 0)

df.head()



df = df.drop("id",1)

df = df.drop("Unnamed: 32",1)

df.diagnosis.unique()

df.head()





d = {'M' : 0, 'B' : 1}

df['diagnosis'] = df['diagnosis'].map(d)

features = list(df.columns[1:31])

print(features)





x = df[features]

y = df["diagnosis"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)



print('justtt xxxx')







Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(X_train,y_train)





accuracy = Tree.score(X_test, y_test)

print(accuracy)



Kfold = KFold(len(df),n_folds = 10,shuffle = False)

print("KfoldCrossVal score using Decision Tree is %s" %cross_val_score(Tree,x,y,cv=10).mean())







Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)





Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())





svc = svm.SVC(kernel='linear',C=1).fit(x,y)

Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc,x,y,cv=10).mean())
from sklearn import tree

from sklearn import svm

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn.linear_model import LogisticRegression







from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



from sklearn.neural_network import MLPClassifier





df = pd.read_csv("../input/data.csv",header = 0)

df.head()



df = df.drop("id",1)

df = df.drop("Unnamed: 32",1)

df.diagnosis.unique()

df.head()





d = {'M' : 0, 'B' : 1}

df['diagnosis'] = df['diagnosis'].map(d)

features = list(df.columns[1:31])





x = df[features]

y = df["diagnosis"]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)



print('justtt xxxx')







Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(X_train,y_train)





accuracy = Tree.score(X_test, y_test)

print(accuracy)



Kfold = KFold(len(df),n_folds = 10,shuffle = False)







Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)





Kfold = KFold(len(df),n_folds=10,shuffle=False)





svc = svm.SVC(kernel='linear',C=1).fit(x,y)

Kfold = KFold(len(df),n_folds=10,shuffle=False)













dt = Tree.fit(X_train,y_train)

y_pred = dt.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy =  dt.score(X_test, y_test)

print('tree')

print(accuracy)







rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy =  rf.score(X_test, y_test)

print('forest')

print(accuracy)





sm = svc.fit(X_train,y_train)

y_pred = sm.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

accuracy = sm.score(X_test, y_test)

print('svm')

print(accuracy)












