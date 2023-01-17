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
train_data = pd.read_csv('../input/train.csv')

print(train_data.head())

print(train_data.describe())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
# Find all the unique genders -- the column appears to contain only male and female.

print(train_data["Sex"].unique())



# Replace all the occurences of male with the number 0.

train_data.loc[train_data["Sex"] == "male", "Sex"] = 0



# Replace all the occurences of female with the number 1.

train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
# Find all the unique values for "Embarked".

print(train_data["Embarked"].unique())



#Fill na Embarked with most common 'S'

train_data['Embarked'] = train_data['Embarked'].fillna('S')



# Replace all occurences of 'S' with 0

train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0



# Replace all occurences of 'C' with 1

train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1



# Replace all occurences of 'Q' with 2

train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2
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

kf = KFold(train_data.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (train_data[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = train_data["Survived"].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(train_data[predictors].iloc[test,:])

    predictions.append(test_predictions)
from sklearn.metrics import accuracy_score



# The predictions are in three separate numpy arrays.  Concatenate them into one.  

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = accuracy_score(train_data['Survived'], predictions)
print(accuracy)
titanic_test = pd.read_csv("../input/test.csv")



# Fillna Age with median

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())



# Fillna Age with median

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())



#Fill na Embarked with most common 'S'

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')



# Replace male occurences of male with 0

titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0



# Replace male occurences of female with 1

titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex']= 1



# Replace male occurences of Embarked 'S' with 0

titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0

# Replace male occurences of Embarked 'C' with 1

titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1

# Replace male occurences of Embarked 'Q' with 2

titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
titanic_test[predictors].isnull().any()
from sklearn.linear_model import LogisticRegression



# Initialize the algorithm class

alg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

alg.fit(train_data[predictors], train_data["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })
submission.to_csv('titanic_submission.csv', index=False)