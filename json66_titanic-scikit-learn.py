# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neural_network import MLPClassifier



train_data = pd.read_csv("../input/train.csv")

train_data.head()
survived = train_data[train_data["Survived"].isin([1])]

survived.head()
# Manipulating Source Data

def process_data(data):

    P_class = data["Pclass"]

    sexes = []



    f_m = ["female", "male"]

    for sex in data["Sex"]:

        sexes.append(f_m.index(sex))



    age = data["Age"]

    age_average = age.mean()



    for i in range(0, len(age)):

        if pd.isnull(age.iat[i]):

            age.iat[i] = age_average



    sibsp = data["SibSp"]

    parch = data["Parch"]

    

    fare = data["Fare"]

    fare_average = fare.mean()

    

    for i in range(0, len(fare)):

        if pd.isnull(fare.iat[i]):

            fare.iat[i] = fare_average



    cabin = []



    cabins = ["T", "A", "B", "C", "D", "E", "F"]

    for c in data["Cabin"]:

        c = str(c)

        if c[0] in cabins:

            cabin.append(cabins.index(c[0]))

        else:

            cabin.append(4)

    

    embark = []

    embarks = ["C", "Q", "S"]

    for e in data["Embarked"]:

        if e in embarks:

            embark.append(embarks.index(e))

        else:

            embark.append(10)



    X = np.stack((P_class, sexes, age, sibsp, parch, fare, cabin, embark), axis=1)

    return X



X = process_data(train_data)

y = train_data["Survived"]



X_df = pd.DataFrame(X,

                 columns=["P_class", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])

X_df.head()
y.head()
clf = MLPClassifier(activation="logistic", solver="lbfgs", alpha=1e-5,

                    hidden_layer_sizes=(7, 6, 2))

clf.fit(X, y)
test_data = pd.read_csv("../input/test.csv")

test_data.head()
test_X = process_data(test_data)

test_X_df = pd.DataFrame(X,

                 columns=["P_class", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])

test_X_df.head()
# Test for missing values in test_X

np.any(np.isnan(test_X))
predictions = pd.Series(clf.predict(test_X))

ids = test_data["PassengerId"]



predictions_df = pd.DataFrame(np.stack([predictions, ids], axis=1),

                              columns=["Survived", "PassengerId"])

predictions_df.head()
# Save output to csv

predictions_df.to_csv("output.csv", index=False)