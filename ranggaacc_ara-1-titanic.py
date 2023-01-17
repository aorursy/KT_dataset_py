

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

cover = pd.read_csv("../input/train.csv")

cover.info()
cover.describe()
cover['Age'].fillna(int(cover.Age.mean()),inplace=True)

#cover.isnull().sum()
cover['Cabin'].fillna(cover.Cabin.mode(),inplace=True)

cover.info()
cover['Age'].fillna(int(cover.Age.mean()),inplace=True)

cover
print(cover['Cabin'].mode())

cover['Cabin'].fillna("B96 B98",inplace=True)

cover.info()
print(cover['Embarked'].mode())

cover['Embarked'].fillna("S",inplace=True)

cover.info()
cover["Class"] = cover["Survived"]

cover.info()

del cover["Survived"]

cover.info()
cover.info()

olist
X = cover[cover.columns[0:11]]

Y = cover["Class"]

olist = list(X.select_dtypes(["object"]))

for col in olist:

    X[col]= X[col].astype("category").cat.codes

del cover["Name"]

X
Y
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

model.fit(X,Y)

model.score(X,Y)
