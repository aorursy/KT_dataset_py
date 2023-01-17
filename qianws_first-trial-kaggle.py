# 2

import pandas as pd



titanic = pd.read_csv('../input/train.csv')



print(titanic.head(5))

print(titanic.describe())



# 3

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())



# 5

# Find all the unique genders -- the column appears to contain only male and female.

print(titanic["Sex"].unique())



# Replace all the occurences of male with the number 0.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



# 6

import numpy as np



print(titanic['Embarked'].unique())

print(titanic['Embarked'].value_counts())

titanic['Embarked'].fillna('S', inplace=True)

embark_dict = {'S': 0, 'C': 1, 'Q': 2}

# titanic['Embarked'] = titanic['Embarked'].apply(lambda x: embark_dict[x])

s = titanic['Embarked'].apply(lambda x: embark_dict[x]).astype(np.int64)

print(s.describe())  # actually np.int64 inside, but describe() fails



titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

s = titanic['Embarked']

print(titanic['Embarked'].describe())



# 9

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

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)  # shuffle=False, random_state is useless



predictions = []

for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train, :])

    # The target we're using to train the algorithm.

    train_target = titanic["Survived"].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test, :])

    predictions.append(test_predictions)



# 10

import numpy as np



# The predictions are in three separate numpy arrays.  Concatenate them into one.

# We concatenate them on axis 0, as they only have one axis.

predictions = np.concatenate(predictions, axis=0)



# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <= .5] = 0



target = titanic['Survived']

cnt_matched = 0

for i in range(target.shape[0]):

    if predictions[i] == target[i]:

        cnt_matched += 1

accuracy = cnt_matched / target.shape[0]

print(accuracy)



# 注意 predictions 不是Series而是ndarray,对它取布尔组下标,行为不同

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)



# 11

from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation



lr = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(lr, titanic[predictors], titanic['Survived'], cv=3)

print(scores.mean())



# 12

import pandas



titanic_test = pandas.read_csv('../input/test.csv')

titanic_test['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test['Embarked'].fillna('S', inplace=True)

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test['Fare'].fillna(titanic_test['Fare'].median(), inplace=True)





# 13

# Initialize the algorithm class

alg = LogisticRegression(random_state=1)



# Train the algorithm using all the training data

alg.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = alg.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pandas.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived": predictions

})

submission.to_csv('./kaggle.csv', index=False)