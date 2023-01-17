#import library for data manipulation and visualization

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Read training & test data from csv files

train_df = pd.read_csv("../input/train.csv")

print(train_df.info())

train_df.head()
test_df = pd.read_csv("../input/test.csv")

print(test_df.info())

test_df.head()
print(train_df.shape)

print(test_df.shape) # test_df doesn't contain a survived feature that is to be predicted.
train_df["Embarked"] = train_df.Embarked.apply(lambda x: x == np.nan and 0 or (x == "S" and 1 or (x == "C" and 2 or 3)))

test_df["Embarked"] = test_df.Embarked.apply(lambda x: x == np.nan and 0 or (x == "S" and 1 or (x == "C" and 2 or 3)))



train_df["Sex"] = train_df.Sex.apply(lambda x: x=="male" and 1 or 0)

test_df["Sex"] = test_df.Sex.apply(lambda x: x=="male" and 1 or 0)
variables = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]

X = train_df[variables]

y = train_df.Survived
# We check for nan and infinity

if not np.isfinite(y).all():

    print("Y contains infinite number")

for var in ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]:

    if not np.isfinite(X[var]).all():

        print(var+" contains infinite number")

        for val in X[var]:

            if not np.isfinite(val):

                print(val)
X_test = test_df[["PassengerId"]+ variables]

X_test = X_test.dropna(how="any")
# Import model classes

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
# search for an optimal value of K for KNN

k_range = list(range(1, 31))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

    k_scores.append(scores.mean())

print(k_scores)
# Plot the different values for k and corresponding Cross Validation accuracy

import matplotlib.pyplot as plt

%matplotlib inline



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')
# 10-fold cross-validation with the best KNN model

knn = KNeighborsClassifier(n_neighbors=7)

print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
# 10-fold cross-validation with logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
logreg.fit(X,y)

y_test = logreg.predict(X_test[variables])

print(y_test.shape)

print(X_test.shape)
R = pd.DataFrame(y_test, index=X_test.PassengerId, columns=["Survived"])

print(R.shape)

R.head()
#R.to_csv("../input/gender_submission.csv")

y_test