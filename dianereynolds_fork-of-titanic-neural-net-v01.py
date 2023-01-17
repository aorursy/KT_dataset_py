import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn import svm

import sklearn as skl

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def clean(mdf):

    mdf = mdf.filter(regex="^(?!Name$).*")

    mdf = mdf.filter(regex="^(?!Cabin$).*")

    mdf = mdf.filter(regex="^(?!Ticket$).*")

    mdf = mdf.filter(regex="^(?!PassengerId$).*")

    mdf.Sex = mdf.Sex.map({'female':1, 'male':0})

    mdf.Embarked = mdf.Embarked.map({'S':1,'C':2,'Q':3})

    mdf = mdf.fillna(0)

    return mdf
test = pd.read_csv("../input/test.csv")

test = clean(test)

#test = testdf.loc[:,'Pclass':'Embarked']



train = pd.read_csv("../input/train.csv")

train = clean(train)



#main = df[:890]

#valid = df[701:]

survive = train.Survived

train.drop('Survived', axis=1, inplace=True)

#survivetest = valid.Survived

#explain = main.loc[:,'Pclass':'Embarked']

#explaintest = valid.loc[:,'Pclass':'Embarked']
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 8, 3), random_state=1)

clf.fit(train, survive)

scores = cross_val_score(clf, train, survive, cv=8)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
csvm = svm.SVC()

csvm.fit(train,survive)

scores = cross_val_score(csvm, train, survive, cv=4)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clsvm = svm.NuSVC()

clsvm.fit(train,survive)

scores = cross_val_score(clsvm, train, survive, cv=4)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clisvm = svm.LinearSVC()

clisvm.fit(train,survive)

scores = cross_val_score(clisvm, train, survive, cv=4)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
csvr = svm.SVR()

csvr.fit(train,survive)

scores = cross_val_score(csvr, train, survive, cv=4)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
csvrn = svm.NuSVR()

csvrn.fit(train,survive)

scores = cross_val_score(csvrn, train, survive, cv=4)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

n_neighbors = 5

knn = skl.neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

knn.fit(train,survive)



#train1 = train[0:880]

#test1 = train[881:]

#survive1 = survive[0:880]

#knn.fit(train1,survive1)

#knn.predict(test1)

#print(survive[881:])
pred = knn.predict(test)

testid = pd.read_csv("../input/test.csv")



subm = pd.DataFrame({

    'PassengerId': testid.PassengerId,

    'Survived': pred

})

subm.to_csv("kaggle.csv", index=False)

#subm
#subm