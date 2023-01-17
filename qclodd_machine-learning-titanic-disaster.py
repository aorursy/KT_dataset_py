import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Loading Training Dataset and Test Dataset

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
# Some NaN values in Age column, Cabin nearly useless (too many NaNs)

train_df.info()
train_df.describe()
# Number of Males and Females on the board

sns.countplot(x="Sex", data=train_df, palette="pastel")
# Did they survived? With distinction between males and females

sns.countplot(x="Survived", hue="Sex", data=train_df, palette="pastel")
# Distributions of Ages - dropped NaNs

sns.distplot(train_df["Age"].dropna(), kde=False, bins=20, color="red")
# Most people Embarked in?

sns.countplot(x="Embarked", data=train_df, palette="pastel")
# Distribution between classes

sns.countplot(x="Pclass", data=train_df, palette="pastel")
train_df.head()
# Preparing Training dataset - Dropping ID, Name, Ticket, Cabin

train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
train_df.head()
# Dummy Pclass

pclass_dummy = pd.get_dummies(train_df["Pclass"], drop_first=True)

pclass_dummy.head()
# Dummy Sex

sex_dummy = pd.get_dummies(train_df["Sex"], drop_first=True)

sex_dummy.head()
# Dummy Embark

embarked_dummy = pd.get_dummies(train_df["Embarked"], drop_first=True)

embarked_dummy.head()
train_df.head()
# Dropping original Pclass, Sex, Embarked columns

train_df.drop(["Pclass", "Sex", "Embarked"],axis=1, inplace=True)
# Concatenating rest of dataset with newly created dummy variables

train_df = pd.concat([train_df, pclass_dummy, sex_dummy, embarked_dummy], axis=1)
train_df.head()
# Replacing NaNs with mean of whole Age column

def age_nan(age_number):

    if np.isnan(age_number):

        return round(np.mean(train_df["Age"]))

    else:

        return age_number



age_none_nan = train_df["Age"].apply(age_nan)
# Rewriting old Age column with Age Series with no NaN values

train_df["Age"] = age_none_nan
# Splitting data into Dependent Variables and Independent Variable

y_train = train_df["Survived"]

X_train = train_df[train_df.columns[1:]]
# Fitting LogisticRegression on Training dataset

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(X_train, y_train)
test_df.head()
# Dropping columns - PassengerId, Name, Ticket, Cabin

test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

test_df.head()
# Dummy variables - Pclass, Sex, Embarked

dummy_pclass = pd.get_dummies(test_df["Pclass"], drop_first=True)

dummy_sex = pd.get_dummies(test_df["Sex"], drop_first=True)

dummy_embarked = pd.get_dummies(test_df["Embarked"], drop_first=True)
# Dropping old columns

test_df.drop(["Pclass", "Sex", "Embarked"], axis=1, inplace=True)
# Concatenating rest of dataset with newly created dummy variables

test_df = pd.concat([test_df, dummy_embarked, dummy_pclass, dummy_sex], axis=1)
test_df.head()
# Removing NaN and replacing with mean of Age column

def age_nan(age_number):

    if np.isnan(age_number):

        return round(np.mean(test_df["Age"]))

    else:

        return age_number



age_none_nan = test_df["Age"].apply(age_nan)

test_df["Age"] = age_none_nan
# Removing NaN and replacing with mean of Fare column

def fare_nan(fare_number):

    if np.isnan(fare_number):

        return round(np.mean(test_df["Fare"]))

    else:

        return fare_number



fare_none_nan = test_df["Fare"].apply(fare_nan)

test_df["Fare"] = fare_none_nan
# Test for other NaN values in Dataframe

test_df.isna().any()
# Make predictions from our model

y_pred = classifier.predict(test_df)
# Graphic results, if passengers from test_df would be dead or not

sns.countplot(y_pred)