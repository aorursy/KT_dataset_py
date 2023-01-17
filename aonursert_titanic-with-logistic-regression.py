import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
# check null values in train and test datasets, later we're going to fill or drop those null values

sns.heatmap(train.isnull())

# or sns.heatmap(pd.isnull(train))
sns.heatmap(test.isnull())
sns.set_style("whitegrid")
# we fill the Age column because it has little amount of null values and those can be filled

# we fill the Age column with relationship between Pclass

# to see this relationship

sns.boxplot(x="Pclass", y="Age", data=train)
sns.boxplot(x="Pclass", y="Age", data=test)
# around age 37 is more likely in class 1, age 29 is in class 2 and, age 24 is in class 3 for train dataset
# for train dataset

def fill_age_train(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train["Age"] = train[["Age", "Pclass"]].apply(fill_age_train, axis=1)
# we can see there is no null values in Age column

sns.heatmap(train.isnull())
# similar things on test dataset

# around age 42 is more likely in class 1, age 26 is in class 2 and, age 24 is in class 3 for train dataset 
# for test dataset

def fill_age_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        elif Pclass == 2:

            return 26

        else:

            return 24

    else:

        return Age
test["Age"] = test[["Age", "Pclass"]].apply(fill_age_test, axis=1)
sns.heatmap(test.isnull())
# Cabin column in both dataset is not fillable because lack of information so we drop it
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
# we can see there is no left null value in general in both dataset
# Sex column values can be either male or female so it can be represent by numerical values

# Same way in Embarked and Pclass columns
sex = pd.get_dummies(train["Sex"], drop_first=True)

embark = pd.get_dummies(train["Embarked"], drop_first=True)

pclass = pd.get_dummies(train["Pclass"], drop_first=True)
train = pd.concat([train, sex, embark, pclass], axis=1)
train.head()
# since we add those columns values with their numerical representations we don't need originial columns so we drop it
# also, some columns can not be converted to numerical so we drop those too
train.drop(["Sex", "Embarked", "Pclass", "Name", "Ticket"], axis=1, inplace=True)
train.head()
# we clean the train dataset, we need to do same processes on test dataset
sex2 = pd.get_dummies(test["Sex"], drop_first=True)

embark2 = pd.get_dummies(test["Embarked"], drop_first=True)

pclass2 = pd.get_dummies(test["Pclass"], drop_first=True)
test = pd.concat([test, sex2, embark2, pclass2], axis=1)
test.head()
test.drop(["Sex", "Embarked", "Pclass", "Name", "Ticket"], axis=1, inplace=True)
test.head()
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
test[test.isnull().any(axis=1)]
test.at[152, 'Fare'] = test["Fare"].mean()
test.at[152, 'Fare']
test[test.isnull().any(axis=1)]
X_train = train.drop("Survived", axis=1)

y_train = train["Survived"]



X_test = test
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.solver = "liblinear"
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
df = pd.DataFrame(data=predictions)
df.columns = ["Survived"]
df
df.index.name = "PassengerId"
df
df.index += 892
df
df.to_csv("my_output")