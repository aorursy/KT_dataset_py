import pandas
# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pandas.read_csv("train.csv")

# Print the first 5 rows of the dataframe.
print(titanic.head(5))
print(titanic.describe())

#there is missing data in the age column. Going to replace gaps with median value for now. 
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print(titanic["Age"].median())

print(titanic["Age"].describe())

titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

print(titanic["Sex"].unique())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna('S')

# Check that there is no more nan values in "Embarked"
print(titanic["Embarked"].unique())

# Change string values of "Embarked" to integers.
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Check that values in "Embarked" are no integers not strings.
print(titanic["Embarked"].unique())

###Now we're going to do some regression. Continuing to use pandas and adding in some scikit-learn.

#there is missing data in the "Embarked" column. Going to replace gaps with most common value for now. 

# This finds the most common value for "Embarked".
print(titanic["Embarked"].mode())
# This replaces the nan with 0, the most common value. I doing it this way for now (instead of leaving out the 2 rows) as I suspect the "embarke doesn't impact much.
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode)

#there is missing data in the "Embarked" column. Going to replace gaps with most common value for now. 

# This finds the most common value for "Embarked".
print(titanic["Embarked"].mode())
# This replaces the nan with 0, the most common value. I doing it this way for now (instead of leaving out the 2 rows) as I suspect the "embarke doesn't impact much.
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode)

titanic = titanic[titanic.Embarked.notnull()]

# trying to figure out why I keep getting this error in next code. ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
print(titanic["Embarked"].describe())

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

print(titanic.loc[61])
print(titanic.loc[832])

import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)


print(accuracy)

from sklearn import cross_validation
from sklearn import linear_model
#sklearn.linear_model.LogisticRegression

# Initialize our algorithm
alg = linear_model.LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
