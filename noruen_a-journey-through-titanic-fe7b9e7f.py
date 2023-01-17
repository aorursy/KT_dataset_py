# pandas

import pandas as pd

from pandas import Series,DataFrame

from sklearn import metrics

from sklearn.model_selection import KFold,cross_val_predict





# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# let's have a look at the data

train.info()
train.describe()
# drop columns that seem not that relevant

train.head()

train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test = test.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
# Embarked

train.info()

test.info()

train.Embarked = train.Embarked.fillna('S')



train.Embarked = train.Embarked.map({'S': 0, 'C':1, 'Q': 2})

test.Embarked = test.Embarked.map({'S': 0, 'C':1, 'Q': 2})
# the fare column also has a NA

train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

train.info()
train.Sex = train.Sex.map({'male': 0, 'female': 1})

test.Sex = test.Sex.map({'male': 0, 'female': 1})
train.describe()

test.describe()
train.Age = train.Age.fillna(train.Age.mean())

test.Age = test.Age.fillna(test.Age.mean())



train.Fare = train.Fare.fillna(train.Fare.mean())

test.Fare = test.Fare.fillna(train.Fare.mean())
# define training and testing sets



X_train = train.drop("Survived",axis=1).drop("PassengerId", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId",axis=1).copy()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Support Vector Machines



# svc = SVC()

# svc.fit(X_train, Y_train)



# Y_pred = svc.predict(X_test)



# svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)

predicted = cross_val_predict(random_forest, train.drop("Survived",axis=1).drop("PassengerId", axis=1),

                              train.Survived, cv=10)

score = metrics.accuracy_score(train.Survived, predicted)

print("score is : %f"%(score))

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

print(Y_pred)
# knn = KNeighborsClassifier(n_neighbors = 3)



# knn.fit(X_train, Y_train)



# Y_pred = knn.predict(X_test)



# knn.score(X_train, Y_train)
# Gaussian Naive Bayes



# gaussian = GaussianNB()



# gaussian.fit(X_train, Y_train)



# Y_pred = gaussian.predict(X_test)



# gaussian.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)