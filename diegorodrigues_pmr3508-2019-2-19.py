import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn import preprocessing
adult = pd.read_csv('C:/Users/Diego/Desktop/Poli/Machine Learnig/all/adult.data.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult.head()
adult.shape
adult["Country"].value_counts()
adult["Age"].plot(kind='hist',bins=15);
adult["Occupation"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="pie",autopct='%.2f')
adult["Target"].value_counts().plot(kind="bar")
nadult = adult.dropna()

nadult.shape
nadult.head()
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult.head()
X_numAdult=numAdult[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Y_numAdult=nadult.Target
adultTest = pd.read_csv('C:/Users/Diego/Desktop/Poli/Machine Learnig/all/adult.test.csv',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adultTest.head()
adultTest.shape
nadultTest = adultTest.dropna()

nadultTest.shape
numAdultTest = nadultTest.apply(preprocessing.LabelEncoder().fit_transform)
X_numAdultTest=numAdultTest[["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Y_numAdultTest=nadultTest.Target
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_numAdult, Y_numAdult)

nb_pred = nb.predict(X_numAdultTest)

print("Gaussian Naive Bayes Metrics")

print(classification_report(Y_numAdultTest, nb_pred))

print('Accuracy: ',accuracy_score(Y_numAdultTest, nb_pred))
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_numAdult,Y_numAdult)

logit_pred = logit.predict(X_numAdultTest)
print("Logistic regression metrics")

print(classification_report(Y_numAdultTest, logit_pred))

print('Accuracy: ',accuracy_score(Y_numAdultTest, logit_pred))
from sklearn.ensemble import RandomForestClassifier

flor = RandomForestClassifier(n_estimators=100)

flor.fit(X_numAdult, Y_numAdult)

flor_pred = flor.predict(X_numAdultTest)
print("Random forest metrics")

print(classification_report(Y_numAdultTest, flor_pred))

print('Accuracy: ',accuracy_score(Y_numAdultTest, flor_pred))