# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Looking at the data

titanic = pd.read_csv("../input/train.csv")

print(titanic.describe())
# Missing data

# Fill missing data with median

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# Check that the Age column has been filled and now has 891 rows

print(titanic.describe())
# Non-numeric columns

# I want to see all the column names, numeric and non-numeric 



print(titanic.head())
# I want to find how many values in the Cabin column are missing

print(titanic["Cabin"].count())
# Converting the Sex column

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
# Checking if "male" has been replaced by 0

print(titanic["Sex"])
# Doing the same for female

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
# Checking female has been replaced

print(titanic["Sex"])
# Converting the Embarked column 

# First how many of the values are "Nan"?

print(titanic["Embarked"].count())

print(titanic["Embarked"].unique())
# 2 values are missing so will convert those to S

# Converting S to 0, C to 1, and Q to 2

# Need to save the results or else the filled values aren't saved



titanic["Embarked"] = titanic["Embarked"].fillna("S")
# Checking that there are no "nan" values any more

print(titanic["Embarked"].unique())
# Converting 

titanic.loc[titanic["Embarked"] == 'S'] = 0

titanic.loc[titanic["Embarked"] == 'C'] = 1

titanic.loc[titanic["Embarked"] == 'Q'] = 2

print(titanic["Embarked"])
# Linear regression for survival predictions

# import linear_model and cross_validation

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold



# columns to use in linear regression prediction

predictors = ["Pclass", "Sex", "Age", "SibSp", "Embarked", "Fare"]

# First I will check how just Age can be used to predict survival

# Logistic regression

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression



# Initialize the algorithm

alg = LogisticRegression(random_state=1)



# Compute accuracy for all the cross validation folds

accuracy = cross_validation.cross_val_score(alg, titanic[predictors], 

                                            titanic["Survived"], cv=3)



# Take the mean of the folds

mean = np.mean(accuracy)



# Now I have to repeat the same procedure on the test set



titanic_test = pd.read_csv("../input/test.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



# Initialize your model

alg = LogisticRegression(random_state = 1)



# Train algorithm on the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Predictions

predictions = alg.predict(titanic_test[predictors])

print (len(predictions))



# Submit file

#submission = pd.DataFrame({

#    "PassengerId":titanic["PassengerId"], 

#    "Survived": predictions

#})



# Generate a csv

#submission.to_csv("kaggle.csv", index=False)




