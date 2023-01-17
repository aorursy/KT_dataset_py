import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

test_survived = pd.read_csv("../input/gender_submission.csv") 

## I prefer merging test and gender_submission, then joining train, later I will split them by using PassengerId.

## By combining data set, we do not need to apply functions to train and test group, separately. Also, filling missing values

## based on combined data set might provide a better result.
train.head()
train.tail()
test.head()
test.tail()
test_survived.head()
test["Survived"] = test_survived["Survived"]
test.head()
## Reordering columns of test similar to train 

test = test[["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]
test.head()
train.head()
## Now, train and test have columns in the same order

titanic = train.append(test, ignore_index= True)
## Now, we have 1308 passengers in the same data frame

titanic.tail()
titanic.info()
titanic.describe()
fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))



sns.countplot(x="Sex", data= titanic, ax= axis1)

sns.countplot(x="Sex", hue= "Survived", data= titanic, ax = axis2)
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (20, 5))

pclass_survived = titanic[["Pclass", "Survived"]].groupby(["Pclass"], as_index= False).mean()



sns.countplot(x= "Pclass", data= titanic, ax= axis1)

sns.barplot(x= "Pclass", y= "Survived", data= titanic, estimator= sum, ax= axis2)

sns.barplot(x= "Pclass", y= "Survived", data= pclass_survived, ax= axis3)
fig, axes = plt.subplots(figsize = (15, 7), ncols= 2, nrows= 2)

sns.countplot(x="SibSp", hue = "Pclass", data= titanic, ax = axes[0, 0])

sns.countplot(x="SibSp", hue = "Survived", data= titanic, ax = axes[0, 1])

sns.countplot(x="Parch", hue = "Pclass", data= titanic, ax = axes[1, 0])

sns.countplot(x="Parch", hue = "Survived", data= titanic, ax = axes[1, 1])

plt.tight_layout
plt.subplots(figsize= (15,6))

sns.violinplot(x= "Pclass", y= "Age", data= titanic, hue= "Survived", split= True)
## Creating a new column for na is Cabin; 1 for na, 0 for non-na in Cabin

titanic["cabin_na"] = np.where(titanic["Cabin"].isnull(), 1, 0)

titanic.head()
## Elaborating Cabin 

plt.subplots(figsize = (10, 5))

sns.countplot(x="Pclass", hue = "cabin_na", data= titanic)
## Fare vs Pclass

plt.subplots(figsize= (20, 8))

sns.boxplot(x="Fare", y= "Pclass", data= titanic, orient="horizontal")
## Checking missing values visually in data set

fig, ax = plt.subplots(figsize=(10,5)) 

sns.heatmap(titanic.isnull(), yticklabels= False, cbar= False, cmap= "PuBu_r", ax = ax)
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x= "Pclass", y= "Age", data= titanic, hue ="Sex", ax = ax)
avg_age_by_Pclass_Sex = pd.pivot_table(data= titanic, values= "Age", index= "Sex", columns="Pclass", aggfunc="mean")

median_age_by_Pclass_Sex = pd.pivot_table(data= titanic, values= "Age", index= "Sex", columns="Pclass", aggfunc="median")

print("Average of Ages")

print(round(avg_age_by_Pclass_Sex))

print("************************")

print("Median of Ages")

print(round(median_age_by_Pclass_Sex))
## Defining function to fill missing ages regarding passenger class and sex

def impute_age(age_age):

    Age = age_age[0]

    Pclass = age_age[1]

    Sex = age_age[2]

    

    if pd.isnull(Age):

        if Pclass == 1:

            if Sex == "female":

                return 37

            else:

                return 41

        elif Pclass == 2:

            if Sex == "female":

                return 27

            else:

                return 31

        else:

            if Sex == "female":

                return 22

            else:

                return 26

    else:

        return Age
## Filling missing ages regarding passenger class and sex

titanic["Age"] = titanic[["Age", "Pclass", "Sex"]].apply(impute_age, axis = 1)
## Check whether there is na in age column

titanic[pd.isnull(titanic["Age"])]
#Elaborating Embarked

embarked_percent_survived = titanic[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()



fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (20, 5))

sns.countplot(x= "Embarked", data= titanic, ax= axis1)

sns.barplot(x= "Embarked", y= "Survived", data= titanic, estimator= sum, ax= axis2)

sns.barplot(x= "Embarked", y= "Survived", data= embarked_percent_survived, ax= axis3, order=["S", "C", "Q"])
## Check whether there is na in embarked column

titanic[pd.isnull(titanic["Embarked"])]
## There are 2 na in Embarked, so we can fill it with S, which is the highest number among S, C and Q

titanic["Embarked"] = titanic["Embarked"].fillna("S")
sex_dummy = pd.get_dummies(titanic["Sex"], drop_first= True)

sex_dummy.head()
embarked_dummy = pd.get_dummies(titanic["Embarked"])
## Sincle Q is the lowest among S, C and Q,we can drop Q

embarked_dummy.drop(["Q"], axis=1, inplace= True)

embarked_dummy.head()
## Changing column names

embarked_dummy.columns = ["embarked_C", "embarked_S"]

embarked_dummy.head()
def age_range(age_age_2):

    Age = age_age_2[0]

    if Age <= 20:

        return "less_than_20"

    elif (Age > 20) & (Age <= 40):

        return "age_between_21_40"

    elif (Age > 40) & (Age <= 60):

        return "age_between_41_60"

    else:

        return "age>60"
titanic["age_range"] = titanic[["Age"]].apply(age_range, axis= 1)
age_range_survived = titanic[["age_range", "Survived"]].groupby("age_range", as_index=False).mean()
plt.subplots(figsize= (10, 5))

sns.barplot(x= "age_range", y="Survived", data= age_range_survived, order = ["less_than_20", "age_between_41_60", "age_between_21_40", "age>60"])
age_range_dummy = pd.get_dummies(titanic["age_range"], drop_first= True)

age_range_dummy.head()
pclass_dummy = pd.get_dummies(titanic["Pclass"], drop_first= True)

pclass_dummy.head()
## Changing column names

pclass_dummy.columns = ["pclass_2", "pclass_3"]

pclass_dummy.head()
titanic_with_dummies = pd.concat([titanic, sex_dummy, embarked_dummy, age_range_dummy, pclass_dummy], axis=1)
titanic_with_dummies.columns.values
## Splitting Titanic data set into original train and test set by using PassengerId (PassengerId 1-886 are train, 

## PassengerId 887-1309)

titanic_train = titanic_with_dummies[titanic_with_dummies["PassengerId"] <= 891]

titanic_test = titanic_with_dummies[titanic_with_dummies["PassengerId"]>=892]
## By X_train, we will predict y_train so that the model is created. Then, we will apply the model to X_test and predict

## survival, which is "predictions". Later, we will compare y_test and predictions.

## Creating y_train and y_test

y_train = titanic_train["Survived"]

y_test = titanic_test["Survived"]
## Dropping irrelevant columns and creating X_train and X_test

X_train = titanic_train.drop(["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Ticket", "Fare", "Cabin", "Embarked", "cabin_na", "age_range"], axis= 1)

X_test = titanic_test.drop(["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Ticket", "Fare", "Cabin", "Embarked", "cabin_na", "age_range"], axis= 1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
logmodel.score(X_train, y_train)
logmodel.score(X_test, y_test)
## Getting output to csv

titanic_result = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })

titanic_result.to_csv('titanic_result.csv', index=False)
titanic_result