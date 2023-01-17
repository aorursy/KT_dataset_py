# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pandas # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

## 2. Looking at the data ##



# We can use the pandas library in python to read in the csv file.

# This creates a pandas dataframe and assigns it to the titanic variable.

titanic = pandas.read_csv("../input/train.csv")



# Print the first 5 rows of the dataframe.

print(titanic.head(5))

print(titanic.describe())



## 3. Missing data ##



# The titanic variable is available here.

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())



## 5. Converting the Sex column ##



# Find all the unique genders -- the column appears to contain only male and female.

print(titanic["Sex"].unique())



# Replace all the occurences of male with the number 0.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



## 6. Converting the Embarked column ##



# Find all the unique values for "Embarked".

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")



titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



## 9. Making predictions ##



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





## 10. Evaluating error ##



import numpy as np



# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)



## 11. Logistic regression ##



from sklearn import cross_validation



# Initialize our algorithm

alg =  xgb.XGBRegressor()

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())





## 12. Processing the test set ##



titanic_test = pandas.read_csv("titanic_test.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")



titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



## 13. Generating a submission file ##



# Initialize the algorithm class

#alg = LogisticRegression(random_state=1)

xgb_model = xgb.XGBRegressor()



alg = GridSearchCV(xgb_model,



                   {'max_depth': [2,4,6],



                    'n_estimators': [50,100,200]}, verbose=1)



# Train the algorithm using all the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pandas.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })
