import numpy as np 

import pandas as pd

from sklearn.linear_model import LogisticRegression
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
def feature(df):

    sex = 1 if df["Sex"] == "male" else 0

    age = df["Age"]

    fare = df["Fare"]

    if np.isnan(age):

        age = round(df_train["Age"].mean())

    if np.isnan(fare):

        fare = df_train["Fare"].mean()

    return [1, age, sex, fare]
Xtrain = df_train.to_dict('records')

Xtrain = [feature(d) for d in Xtrain]

ytrain = df_train["Survived"].to_list()



Xval = Xtrain[700:]

yval = ytrain[700:]

Xtrain = Xtrain[:700]

ytrain = ytrain[:700]
model = LogisticRegression(fit_intercept=False, random_state=42, solver="lbfgs")

model.fit(Xtrain, ytrain)
val_pred = model.predict(Xval)

correct = val_pred == np.array(yval)

acc = sum(correct) / len(correct)

print("Accuracy {}".format(acc))
Xtest = df_test.to_dict('records')

Xtest = [feature(d) for d in Xtest]

test_pred = model.predict(Xtest)
ids = df_test["PassengerId"].to_list()

file = open("answer.csv", "w")

file.write("PassengerId,Survived\n")

for id_, pred in zip(ids, test_pred):

    file.write("{},{}\n".format(id_, pred))

file.close()