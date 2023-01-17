# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# use pandas library to read in csv file. Creates pandas dataframe

titanic = pd.read_csv("../input/train.csv")



#print the first 5 rows of the dataframe

print(titanic.head(5))



print("-------")

# print description of dataset

print(titanic.describe())
# age column has count 714, others have 891. Missing values!

# assign missing values to median age

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# can't use non-numeric columns in manchine learning.

# non-numeric columns: Name, Sex, Cabin, Embarked, Ticket

# ignore Ticket, Cabin and Name

# cabin column is mostly missing (204 our of 891)

# Ticket unlikely to tell much without knowledge of boat

# Name don't know how name correlates with wealth etc.

titanic = titanic.drop(["Name", "Ticket", "Cabin"], axis=1)



# convert Sex column to numeric column

# male = 0, female = 1 

# select male values in sex column and replace with 0

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0



# replace female values with 1

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# now convert Embarked column to codes

# Embarked values are S, C, Q and nan

print(titanic["Embarked"].unique())
# Replace missing values with most common embarkation port, S

titanic["Embarked"] = titanic["Embarked"].fillna("S")



# assign code 0 to S, 1 to C and 2 to Q

# Replace in Embarked column

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic.info()
# import the linear regression class

from sklearn.linear_model import LinearRegression

# import cross validation helper

from sklearn.model_selection import KFold



# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Initialize our algorithm class

alg = LinearRegression()



# Generate cross validation folds for the titanic dataset.  

# It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits 

# every time we run this.

kf = KFold(n_splits=3, random_state=1)

kf.get_n_splits(titanic.shape[0])

KFold(n_splits=3, random_state=1, shuffle=True)



predictions = []



for train, test in kf.split(titanic):

    # The predictors we're using the train the algorithm.  

    # Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train,:])

    

    # The target we're using to train the algorithm.

    train_target = titanic["Survived"].iloc[train]

    

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)

    
# combine 3 sets of predictions into one column

# predictions = np.concatenate(predictions, axis=0)



# map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <= .5] = 0



# check accuracy

acc=0

for i in range(len(predictions)):

    if (predictions[i] == titanic["Survived"][i]):

        acc += 1

accuracy = acc/len(predictions)

print(accuracy)
# import the logistic regression class

from sklearn.linear_model import LogisticRegression



from sklearn import cross_validation



# initialise algorithm

alg = LogisticRegression(random_state=1)



# compute accuracy score for all the cross validations folds

scores = cross_validation.cross_val_score(\

                                          alg,\

                                          titanic[predictors],\

                                         titanic["Survived"],\

                                         cv=3)

# take the mean of the scores

mean_scores = scores.mean()

print(mean_scores)
titanic_test = pd.read_csv("../input/test.csv")



# process test data the same way as processed training data

# Age (use training data to find the median)

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())



# Sex

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1



# Embarked

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



# Fare

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
# Generate a submission file

# first train algorithm on training data, then make predictions

# on test set, then generate a new scv file



# initialise the algorithm

alg = LogisticRegression(random_state=1)



# train the algorithm using the training data

alg.fit(titanic[predictors], titanic["Survived"])



# make predictions using the test data

predictions = alg.predict(titanic_test[predictors])



# create a new dataframe with only the passengerID and Survived

submission = pd.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived": predictions

})



# create output file

results = submission.to_csv("results.csv", index=False)