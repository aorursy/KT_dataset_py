import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import os

pd.set_option("display.max_columns", 25)
train = pd.read_csv("../input/train.csv", index_col="PassengerId")

test = pd.read_csv("../input/test.csv", index_col = "PassengerId")

example = pd.read_csv("../input/gender_submission.csv", index_col = "PassengerId")
def initial_explore(df):

    print(df.describe(), "\n")

    print(df.isna().sum(), "\n")

    print(df.dtypes)
initial_explore(train)
initial_explore(test)
#AGE

plt.hist(train.loc[train["Survived"] == 1]["Age"], bins=15, color = "g", alpha = 0.8)

plt.hist(train.loc[train["Survived"] == 0]["Age"], bins = 15, color = "r", alpha = 0.3)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution on the titanic")

plt.show()
#SEX

sns.countplot("Sex", data = train, hue = "Survived")

plt.xlabel("Gender")

plt.ylabel("Frequency")

plt.title("Gender distribution on the titanic")

plt.show()
#EMBARK LOCATION

sns.countplot("Embarked", data = train, hue = "Survived")

plt.xlabel("Location Embarked")

plt.ylabel("Frequency")

plt.title("Distribution of embark locations on the titanic")

plt.show()
#CLASS

sns.countplot("Pclass", data = train, hue = "Survived")

plt.xlabel("Class")

plt.ylabel("Frequency")

plt.title("Class distribution on the titanic")

plt.show()
def model_prep(df):

    df = pd.get_dummies(df, columns = ["Embarked", "Pclass"])

    df = df.replace("male", 0).replace("female", 1)

    df["Age"].fillna(28, inplace = True) #median age train data

    

    return df



train = model_prep(train)
X = train[["Pclass_1", "Pclass_2", "Pclass_3", "Sex", "Age", "Embarked_C", "Embarked_Q", "Embarked_S"]]

y = train["Survived"]



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 42)



model = LogisticRegression(solver = "liblinear")

model.fit(Xtrain, ytrain)

print("Model score:", model.score(Xtrain, ytrain))



ypred = model.predict(Xtest)



print("Accuracy:", accuracy_score(ytest, ypred))

print("Precision:", precision_score(ytest, ypred))

print("Recall:", recall_score(ytest, ypred))

print("F1-score:", f1_score(ytest, ypred))
test2 = model_prep(test)

test2_X = test2[["Pclass_1", "Pclass_2", "Pclass_3", "Sex", "Age", "Embarked_C", "Embarked_Q", "Embarked_S"]]

predictions = model.predict(test2_X)

result = pd.DataFrame()

result["PassengerId"] = test2_X.index

result["Survived"] = predictions



result.to_csv("Titanic_submission.csv", index = False)