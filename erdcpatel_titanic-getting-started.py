# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline



# machine learning

# Import the linear regression class

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

# Sklearn also has a helper that makes it easy to do cross-validation

from sklearn.model_selection import KFold
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

titanic_test    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()
titanic_df.info()

print("----------------------------")

titanic_test.info()
print(titanic_df.describe())

print('--------')

print(titanic_test.describe())
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
print(titanic_df["Sex"].unique())
titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0

titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1
print(titanic_df["Embarked"].unique())
titanic_df["Embarked"] = titanic_df["Embarked"].fillna('S')
titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0

titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1

titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2
# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm class

lr = LinearRegression()

# Generate cross-validation folds for the titanic data set

# It returns the row indices corresponding to train and test

# We set random_state to ensure we get the same splits every time we run this

kf = KFold(3, random_state=1)



predictions = []

for train, test in kf.split(titanic_df):

    # The predictors we're using to train the algorithm  

    # Note how we only take the rows in the train folds

    train_predictors = (titanic_df[predictors].iloc[train,:])

    # The target we're using to train the algorithm

    train_target = titanic_df["Survived"].iloc[train]

    # Training the algorithm using the predictors and target

    lr.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = lr.predict(titanic_df[predictors].iloc[test,:])

    predictions.append(test_predictions)
# The predictions are in three separate NumPy arrays  

# Concatenate them into a single array, along the axis 0 (the only 1 axis) 

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (the only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = len(predictions[predictions == titanic_df["Survived"]]) / len(predictions)
print(accuracy)
from sklearn.model_selection import cross_val_score



# Initialize our algorithm

alg = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before

scores = cross_val_score(alg, titanic_df[predictors], titanic_df["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_df["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
# Initialize the algorithm class

alg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

alg.fit(titanic_df[predictors], titanic_df["Survived"])



# Make predictions using the test set

predictions = alg.predict(titanic_test[predictors])
# Create a new dataframe with only the columns Kaggle wants from the data set

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('titanic.csv', index=False)