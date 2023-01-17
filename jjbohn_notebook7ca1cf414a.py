import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as P



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/train.csv", header=0)
df[df["Age"] > 60][["Sex", "Pclass", "Age", "Survived"]]
df.groupby(["Pclass", "Sex"])[["Sex"]].count()["Sex"].plot(kind="bar")
df["Age"].dropna().hist(bins=16, range=(0, 80), alpha=.5)

P.show()
df["Gender"] = 4

df["Gender"] = df["Sex"].map({"female": 0, "male": 1}).astype(int)
df["Embarked"] = df["Embarked"].fillna("S")
median_ages = np.zeros((2, 3))



for i in range(0, 2):

    for j in range(0, 3):

        by_gender_and_class = df[(df['Gender'] == i) & (df['Pclass'] == j +1 )]

        median_ages[i,j] = by_gender_and_class["Age"].dropna().median()

 

median_ages
df["AgeFill"] = df["Age"]

df[df["Age"].isnull()][["Gender", "Pclass", "Age", "AgeFill"]].head()



for i in range(0, 2):

    for j in range(0, 3):

        df.loc[(df["Age"].isnull()) & (df["Gender"] == i) & (df["Pclass"] == j + 1), "AgeFill"] = median_ages[i, j]
from sklearn.svm import SVC

test = pd.read_csv("../input/test.csv", header=0)

print(test.info())

features_train = df["AgeFill"].reshape(-1, 1)

labels_train = df["Survived"]



test["Age"] = test["Age"].fillna(29)



clf = SVC()

clf.fit(features_train, labels_train)

pred = clf.predict(test["Age"].reshape(-1, 1))
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred

    })



submission.to_csv('titanic.csv', index=False)