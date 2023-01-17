import numpy as np

import pandas as pd

import seaborn as sns

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix  

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from math import sqrt

from sklearn.metrics import mean_squared_error
train = pd.read_csv("../input/eval-lab-1-f464-v2/train.csv")

train.dropna(axis = 0, inplace=True)

train["type"] = train.type.eq("new").mul(1)
train.head()
test = pd.read_csv("../input/eval-lab-1-f464-v2/test.csv")

test.fillna(test.mean(), inplace = True)

test["type"] = test.type.eq("new").mul(1)

testX = test.drop(["id",], axis = 1)
test.head()
from sklearn.model_selection import train_test_split

X = train.drop(["id", "rating"], axis = 1)

Y = train["rating"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
# from sklearn.preprocessing import StandardScaler



# numerical_feature = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11"]



# x_train[numerical_feature] = StandardScaler().fit_transform(x_train[numerical_feature])

# x_test[numerical_feature] = StandardScaler().fit_transform(x_test[numerical_feature])

# x_train[numerical_feature].describe()
# from sklearn.preprocessing import RobustScaler



# numerical_feature = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "type", "feature10", "feature11"]



# scaler = RobustScaler()

# x_train[numerical_feature] = scaler.fit_transform(x_train[numerical_feature])

# x_test[numerical_feature] = scaler.transform(x_test[numerical_feature])



# x_train[numerical_feature].head()
from sklearn.ensemble import RandomForestRegressor

clf=RandomForestRegressor(n_estimators=500, random_state=42, max_depth=50)

clf.fit(x_train, y_train)
rdfpredreg = clf.predict(x_test)
for i in range(len(rdfpredreg)):

    rdfpredreg[i] = round(rdfpredreg[i])
rdfpredreg = rdfpredreg.astype(np.int64)
accuracy_score(y_test, rdfpredreg)
sqrt(mean_squared_error(y_test, rdfpredreg))
clf=RandomForestRegressor(n_estimators=500, random_state=42, max_depth=50)

clf.fit(X, Y)
rdfpredregFinal = clf.predict(testX)
rdfpredregFinal
for i in range(len(rdfpredregFinal)):

    rdfpredregFinal[i] = round(rdfpredregFinal[i])
rdfpredregFinal = rdfpredregFinal.astype(np.int64)
testId = test["id"]

a = list(zip(["id",], ["rating",]))

a = a + (list(zip(testId, rdfpredregFinal)))

for i in range(len(a)):

    a[i] = list(a[i])
finaldf = pd.DataFrame(data=a[1:][0:], columns=a[0][0:])

finaldf.to_csv('rdfreg.csv', index = False)
from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=500, random_state=42, max_depth=50)

clf.fit(x_train, y_train)
etrpred = clf.predict(x_test)
for i in range(len(etrpred)):

    etrpred[i] = round(etrpred[i])
accuracy_score(y_test, etrpred)
sqrt(mean_squared_error(y_test, etrpred))
# scaler = StandardScaler()

# X[numerical_feature] = scaler.fit_transform(X[numerical_feature])



# X[numerical_feature].head()
clf = ExtraTreesRegressor(n_estimators=500, random_state=42, max_depth=50)

clf.fit(X, Y)
# scaler = StandardScaler()

# testX[numerical_feature] = scaler.fit_transform(testX[numerical_feature])



# testX[numerical_feature].head()
etrFinal = clf.predict(testX)
len(etrFinal)
type(etrFinal[0])
for i in range(len(etrFinal)):

    etrFinal[i] = round(etrFinal[i])
etrFinal = etrFinal.astype(np.int64)
testId = test["id"]

a2 = list(zip(["id",], ["rating",]))

a2 = a2 + (list(zip(testId, etrFinal)))

for i in range(len(a2)):

    a2[i] = list(a2[i])
finaldf2 = pd.DataFrame(data=a2[1:][0:], columns=a2[0][0:])
finaldf2.to_csv('rdfETR.csv', index = False)