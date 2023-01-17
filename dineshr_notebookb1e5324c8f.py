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
from sklearn.linear_model import LinearRegression

from sklearn import cross_validation

from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression 







titanic = pd.read_csv("../input/train.csv")



# The titanic variable is available here.

titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())





# Find all the unique genders -- the column appears to contain only male and female.

print(titanic["Sex"].unique())



# Replace all the occurences of male with the number 0.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"]=="female","Sex"]=1





# Find all the unique values for "Embarked".

print(titanic["Embarked"].unique())





titanic["Embarked"]=titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"]=="S","Embarked"]=0

titanic.loc[titanic["Embarked"]=="C","Embarked"]=1

titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2





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

	

	

	

	





# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)	

print(accuracy)
titanic_test = pd.read_csv("../input/test.csv")



#replace missing value for Age in test set

titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())



#Assign numbers to "sex"

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1



#Assign "S" to missing embarked values 

titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")



#Assign numbers to embarked values 

titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"]=0

titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"]=1

titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"]=2



#Assign median value to misisng Fare

titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())



# Initialize the algorithm class

alg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("submission.csv", index=False)



print(submission)



# Tutorial 1: ADOPTED FROM Gender Based Model (0.76555) - by Myles O'Neill

#https://www.kaggle.com/mylesoneill/titanic/tutorial-part-1-naive-gender-prediction

import numpy as np

import pandas as pd

import pylab as plt



# (1) Import the Data into the Script

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# (2) Create the submission file with passengerIDs from the test file

submission_naive = pd.DataFrame({"PassengerId": 

    test['PassengerId'], "Survived": pd.Series(dtype='int32')})





# (3) Assume that everyone died. 

submission_naive.Survived = 0





# (4) Create a new submission based on gender. 

submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})



# (5) Fill the Data for the survived column, all females live (1) all males die (0)

submission.Survived = [1 if x == 'female' else 0 for x in test['Sex']]



# (6) Create final submission file

submission_naive.to_csv("submission_naive.csv", index=False)

submission.to_csv('submission_gender.csv', index=False)



submission.head()
