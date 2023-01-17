import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pylab as pl

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

print("All libraries have been Imported")
url = '../input/titanic/train.csv'

dft = pd.read_csv(url)

dft.head()
import seaborn as sns 

sns.countplot(x = 'Embarked', data = dft, hue = 'Sex', color = 'green')
dft["Age"]= dft["Age"].fillna(dft["Age"].mean())
dummy = pd.get_dummies(dft["Sex"])

dummy.head()
dft2 = pd.concat((dft, dummy), axis=1)

dft2.head()
dft2.drop(["male"], axis = 1)
dft2 = dft2.drop(["Sex"], axis =1)
dft2 = dft2.rename(columns={"female":"Sex"})
dft2 = dft2.drop(["male"], axis = 1)

dft2.head()
dft2["Embarked"].value_counts()
dft2["Embarked"].isnull().value_counts()
dummy2 = pd.get_dummies(dft2["Embarked"], prefix="Embarked").iloc[:,1:]

dummy2
dft2= dft2.drop(columns={"Embarked"})

dft2.head()
dft2 = pd.concat((dft2,dummy2), axis =1)

dft2
# Visualise the numerical values

dft2.hist(figsize=(12,12), color='green', bins=25)

plt.show()
url2 = '../input/titanic/test.csv'

dftest = pd.read_csv(url2)

dftest.head()
print(dftest.shape)

print(dft2.shape)
dftest["Age"]= dftest["Age"].fillna(dftest["Age"].mean())
dummy3 = pd.get_dummies(dftest["Sex"])

dummy3.head()
dftest = pd.concat((dftest, dummy3), axis=1)

dftest.head()
dftest.drop(["male"], axis = 1)
dftest = dftest.drop(["Sex"], axis =1)
dftest = dftest.rename(columns={"female":"Sex"})
dftest = dftest.drop(["male"], axis = 1)

dftest.head()
dftest[["Embarked","Sex"]].value_counts().plot(kind = 'bar',color = 'blue', figsize=(7,5))

plt.show()
dftest["Embarked"].isnull().value_counts()
dummy4 = pd.get_dummies(dftest["Embarked"], prefix="Embarked").iloc[:,1:]

dummy4
dftest= dftest.drop(columns={"Embarked"})

dftest.head()
dftest = pd.concat((dftest,dummy4), axis =1)

dftest
dft2.Fare.isnull().value_counts()
dftest["Fare"]= dftest["Fare"].fillna(dftest["Fare"].mean())
X = np.asarray(dft2[["Age", "Fare", "Sex","Pclass"]])

X[0:5]

y = np.asarray(dft2["Survived"])

y[0:5]
X = preprocessing.StandardScaler(). fit(X).transform(X)

X[0:5]
X_train = X

y_train = y
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, y_train)

y_pred = LR.predict(X_train)

acc_log = round(LR.score(X_train, y_train) * 100, 2)

acc_log
from sklearn.linear_model import LinearRegression

LinR = LinearRegression()

LinR.fit(X_train, y_train)

y_pred = LinR.predict(X_train)

acc_lin = round(LR.score(X_train, y_train) * 100, 2)

acc_lin
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_train)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



y_pred = linear_svc.predict(X_train)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
results = pd.DataFrame({

    'Model': ['Logistic Regression','Linear Regression', 'Random Forrest', 'SVM'],

    'Score': [acc_log,acc_lin,acc_random_forest,acc_linear_svc]})

    

results_df = results.sort_values(by='Score', ascending = False)

results_df = results_df.set_index('Score')

results_df.head()
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, y_train, cv=15, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':dft2[["Age", "Fare", "Sex","Pclass"]],'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(6)
importances.plot.bar(color='green')
print("oob score:", round(random_forest.oob_score, 4)*100, "%")
X2 = np.asarray(dftest[["Age", "Fare", "Sex","Pclass"]])

X[0:5]



X2 = preprocessing.StandardScaler(). fit(X2).transform(X2)

X2[0:5]



X2_test = X2
testPredictionsRF = random_forest.predict(X2_test)

testPredictionsRF
outputRF = pd.DataFrame({'PassengerId': dftest.PassengerId, 'Survived': testPredictionsRF})

outputRF.to_csv('submission_hanan_1.csv', index=False)