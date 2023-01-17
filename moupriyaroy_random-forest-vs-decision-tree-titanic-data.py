import numpy as np

import pandas as pd
myData = "/kaggle/input/titanic/titanic.csv"

titanic = pd.read_csv(myData)

print(titanic.shape)

titanic.head()
X = titanic.iloc[:,0:titanic.shape[1]-1]

Y = titanic.iloc[:,-1]

Y
from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=123211)

print(type(X_train))

print(type(Y_train))

print(type(Y_test))
X_Y_train = X_train.copy()

X_Y_train['Survived'] = Y_train

print(X_Y_train.shape)

print(X_test.shape)

print(Y_train.shape)

X_Y_train.to_csv('titanic_x_y_train.csv', index=False)

X_train.to_csv('titanic_x_train.csv', index=False)

Y_train.to_csv('titanic_y_train.csv', index=False)

X_test.to_csv('titanic_x_test.csv', index=False)

Y_test.to_csv('titanic_y_test.csv', index=False)
X_Y_train_l = pd.read_csv('titanic_x_y_train.csv', delimiter=',')

X_test_l = pd.read_csv('titanic_x_test.csv', delimiter=',')

print(X_Y_train_l.shape)

X_Y_train_l.isnull().sum()



#X_Y_train_l.head()
del X_Y_train_l["Name"]

del X_Y_train_l["Ticket"]

del X_Y_train_l["Cabin"]

del X_Y_train_l["Fare"]

X_Y_train_l.head()
del X_test_l["Name"]

del X_test_l["Ticket"]

del X_test_l["Cabin"]

del X_test_l["Fare"]
print(X_Y_train_l.isnull().sum())

print(X_Y_train_l.shape)

#X_Y_train_l.dropna(inplace=True)

X_Y_train_l = X_Y_train_l.fillna(X_Y_train_l.median())

print(X_Y_train_l.isnull().sum())

X_Y_train_l.shape



X_Y_train_l.dropna(inplace=True)

print(X_Y_train_l.isnull().sum())

X_Y_train_l.shape
X_train_l = X_Y_train_l.iloc[:,:X_Y_train_l.shape[1]-1]

print(X_train_l.shape)

Y_train_l = X_Y_train_l.iloc[:,-1]

print(Y_train_l.shape)

print(Y_train_l)
def getNumber(str):

    if str=="male":

        return 1

    else:

        return 2

X_train_l['gender']=X_train_l["Sex"].apply(getNumber)

print(X_train_l)

del X_train_l["Sex"]

X_train_l.head()
X_test_l['gender']=X_test_l["Sex"].apply(getNumber)

print(X_test_l)

del X_test_l["Sex"]

X_test_l.head()
def getNumberEmbarked(str):

    if str=='C':

        return 1

    elif str=='Q':

        return 2

    else:

        return 3

X_train_l['Embarked']=X_train_l["Embarked"].apply(getNumberEmbarked)

print(X_train_l)
X_test_l['Embarked']=X_test_l["Embarked"].apply(getNumberEmbarked)

print(X_test_l)
print(X_test_l.isnull().sum())

X_test_l = X_test_l.fillna(X_test_l.median())

X_test_l.isnull().sum()
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf.fit(X_train_l, Y_train_l)

Y_pred_l = clf.predict(X_test_l)

clf.score(X_train_l,Y_train_l), clf.score(X_test_l,Y_test)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 6, random_state=0)

clf.fit(X_train_l, Y_train_l)

clf.score(X_train_l,Y_train_l),clf.score(X_test_l,Y_test)