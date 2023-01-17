import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt
train_adult = pd.read_csv("/kaggle/input/adult-data-5/train_data.csv" ,names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

test_adult =  pd.read_csv("/kaggle/input/adult-data-5/test_data.csv" ,names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
train_adult.shape
test_adult.shape
train_adult.head()
train_adult["Education"].value_counts()
train_adult["Relationship"].value_counts().plot(kind="bar")
ntrain = train_adult.dropna()

ntrain
ntest = test_adult.dropna()

ntest
xtrain = ntrain[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

ytrain = ntrain.Target
xtest = ntest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

ytest = ntest.Target
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ytrain = le.fit_transform(ytrain)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
rf = RandomForestClassifier(n_estimators = 800)
%%time

rf.fit(xtrain, ytrain)
%%time

scores = cross_val_score(rf, xtrain, ytrain, cv=6)

print(scores)
rf = RandomForestClassifier(n_estimators = 80)
%%time

rf.fit(xtrain, ytrain);
%%time

scores = cross_val_score(rf, xtrain, ytrain, cv=8)

print(scores)
numtrain = ntrain.apply(preprocessing.LabelEncoder().fit_transform)

numtest = ntest.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = numtrain.iloc[:,0:14]

Xtest = numtest.iloc[:,0:14]
Ytrain = numtrain.Target

Ytest = numtest.Target
%%time

rf.fit(Xtrain, Ytrain)

YtestPred = rf.predict(Xtest);
print(confusion_matrix(Ytest,YtestPred))
accuracy_score(Ytest, YtestPred)
print(classification_report(Ytest, YtestPred))
from sklearn.svm import SVC
svc = SVC(gamma='auto', cache_size=7000)  # cache_size=7000 decreases execution time
%%time

svc.fit(xtrain, ytrain)
%%time

scores = cross_val_score(svc, xtrain, ytrain, cv=5, n_jobs=3)

print(scores)
from sklearn.decomposition import PCA

# Make an instance of the Model

pca = PCA(.95)  # chooses the minimum number of principal components such that 95% of the variance is retained.
pca.fit(Xtrain)
Newtrain = pca.transform(Xtrain)

Newtest = pca.transform(Xtest)
%%time

svc.fit(Newtrain, Ytrain)
%%time

scores = cross_val_score(svc, Newtrain, Ytrain, cv=5, n_jobs=3)

print(scores)
%%time

YtestPred = svc.predict(Newtest)
print(confusion_matrix(Ytest,YtestPred))
accuracy_score(Ytest, YtestPred)
print(classification_report(Ytest, YtestPred))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
%%time

nb.fit(xtrain, ytrain)
%%time

scores = cross_val_score(nb, xtrain, ytrain, cv=10)

print(scores)
%%time

nb.fit(Xtrain, Ytrain)

YtestPred = nb.predict(Xtest);
print(confusion_matrix(Ytest,YtestPred))
accuracy_score(Ytest, YtestPred)
print(classification_report(Ytest, YtestPred))