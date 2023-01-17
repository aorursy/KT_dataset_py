# ## messing about with _seaborn_
#
# let's prepare data below, then analyze
#
import numpy as np
import pandas as pd

import seaborn as sns
#%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#load & clean from @omarelgabry
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)

#impute median / mode values
for dataframe in [train, test]:
    dataframe["Embarked"] = dataframe["Embarked"].fillna("S")
    dataframe["Fare"].fillna(dataframe["Fare"].median(), inplace=True)
    dataframe["Age"].fillna(dataframe["Age"].median(), inplace=True)
    dataframe['HasCabin'] = dataframe["Cabin"]
    dataframe["HasCabin"].fillna(0, inplace=True)
    dataframe["HasCabin"].loc[dataframe["HasCabin"] != 0] = 1
    dataframe.drop("Cabin", axis=1, inplace=1)
    dataframe['Family'] =  dataframe["Parch"] + dataframe["SibSp"]

print("\n\nSummary statistics of training data")

print("Train --> \n")
print(train.head())
print(train.describe())
print("\n\n")

print("Test --> \n")
print(test.head())
print(test.describe())
print("\n\n")

sns.set_palette('pastel')


sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3,kind="bar")

sns.lmplot(x='Fare', y='Survived', data=train, size=7, logistic=True)
sns.lmplot(x='Pclass', y='Survived', data=train, size=7, logistic=True)
sns.lmplot(x='Age', y='Survived', data=train, size=7, logistic=True)
sns.lmplot(x='SibSp', y='Survived', data=train, size=7, logistic=True)
sns.lmplot(x='Parch', y='Survived', data=train, size=7, logistic=True)
sns.lmplot(x='Family', y='Survived', data=train, size=7, logistic=True)
# clean up categorical variables & prep for random forest model
# undesired variables
X_train = train.copy()
X_test = test.drop("PassengerId",axis=1).copy()

Y_train = train["Survived"]
X_train.drop("Survived", axis=1, inplace=True)

print(X_train.head())
print(Y_train.head())
print(X_test.head())
# one-hot encoding
#below code is slightly slower than necessary - iterates twice per one-hot encoding

for dataframe in [X_train, X_test]:
    dataframe['IsFemale'] = dataframe["Sex"]
    dataframe["IsFemale"].loc[dataframe["IsFemale"] == "female"] = 1
    dataframe["IsFemale"].loc[dataframe["IsFemale"] == "male"] = 0

    dataframe['IsFirst'] = dataframe["Pclass"]
    dataframe["IsFirst"].loc[dataframe["IsFirst"] != 1] = 0
    dataframe["IsFirst"].loc[dataframe["IsFirst"] == 1] = 1
    dataframe['IsSecond'] = dataframe["Pclass"]
    dataframe["IsSecond"].loc[dataframe["IsSecond"] != 2] = 0
    dataframe["IsSecond"].loc[dataframe["IsSecond"] == 2] = 1

    #just checking embarked S for now since it's the largest effect; should add embarked C or Q
    dataframe['EmbarkedS'] = dataframe["Embarked"]
    dataframe["EmbarkedS"].loc[dataframe["EmbarkedS"] != "S"] = 0
    dataframe["EmbarkedS"].loc[dataframe["EmbarkedS"] == "S"] = 1

    dataframe.drop('Embarked', axis=1, inplace=True)
    dataframe.drop('Pclass', axis=1, inplace=True)
    dataframe.drop('Sex', axis=1, inplace=True)

    #also add IsChild on age <=16

print(list(X_train.columns.values))
print(list(X_test.columns.values))

print(X_train.head())
print(Y_train.head())
print(X_test.head())
# Random Forests

random_forest = RandomForestClassifier(n_estimators=300)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
# Write submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_RF.csv', index=False)