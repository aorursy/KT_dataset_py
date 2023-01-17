import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import array
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
train = pd.read_csv ("../input/titanic/train.csv")
train.head()
train.info()
del train['Cabin']
train=train.dropna(subset = ['Age', 'Embarked'])
train.info()
train = train.drop (columns=['Name'])
train = train.drop (columns=['PassengerId'])
train = train.drop (columns=['Ticket'])
train.info()
from IPython.display import Image
Image(filename='../input/titanic-visualization/Count of People Survived.png')
Image(filename='../input/titanic-visualization/Count of People Survived based on Sex.png')
Image(filename='../input/titanic-visualization/Count of People Survived based on Embarked.png')
train_dummies = pd.get_dummies (train)
train_dummies.info()
X = train_dummies.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values
Y = train_dummies.iloc[:, [0]].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
#Classification Using Random Forest
classifier = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', max_depth = 6)
classifier.fit(X_Train,Y_Train.ravel())
Y_Pred = classifier.predict(X_Test)
Y_Pred
cm = confusion_matrix(Y_Test, Y_Pred)
cm
test = pd.read_csv ("../input/titanic-testing-set-without-missing-values/test.csv")
test = test.drop (columns=['Name'])
test = test.drop (columns=['PassengerId'])
test = test.drop (columns=['Ticket'])
test = test.drop (columns=['Cabin'])
test.info()
test_dummies = pd.get_dummies (test)
test_dummies.info()
X = test_dummies.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
test1 = pd.read_csv ("../input/titanic/gender_submission.csv")
Y = test1.iloc[:, [1]].values
sc_X = StandardScaler()
X_Test = sc_X.fit_transform(X)
Y_Pred = classifier.predict(X_Test)
cm = confusion_matrix(Y, Y_Pred)
cm




