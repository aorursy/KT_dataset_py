import numpy as np

import pandas as pd



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



pd.options.mode.chained_assignment = None  # disable the chain assignment warning
# Load the train and test datasets into two DataFrames

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# view the dataset

train.head()
# Description of the dataset

train.describe()
# Check for survival rate based on certain features



# Passengers' survival

print('Passenger:')

print(train['Survived'].value_counts(normalize=True))



# Males' survival

print('\nMale:')

print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True))



# Females' survival

print('\nFemale:')

print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True))
# Check for children



# create column

train['Child'] = np.NaN



# assign values

# 1) child: age < 18

# 2) adult: age >= 18

train['Child'][train['Age'] < 18] = 1

train['Child'][train['Age'] >= 18] = 0



# Child survival

print('Children:')

print(train['Survived'][train['Child'] == 1].value_counts(normalize=True))



# Adult survival

print('\nAdult:')

print(train['Survived'][train['Child'] == 0].value_counts(normalize=True))
# Cleaning and formatting the training data



# Convert gender to binary

# Male: 0

# Female: 1

train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1



# Fill the NaN values of Emabrked with most common value

train['Embarked'] = train['Embarked'].fillna(max(train['Embarked'].value_counts().index))



# Convert the Embarked classes to integer form

train['Embarked'][train['Embarked'] == "S"] = 0

train['Embarked'][train['Embarked'] == "C"] = 1

train['Embarked'][train['Embarked'] == "Q"] = 2



# Fill the NaN values of Age and Child with median value

train['Age'] = train['Age'].fillna(train['Age'].median())

train['Child'] = train['Child'].fillna(train['Child'].median())
# View the test data

test.head()
# Describe the test set

test.describe()
# Filling up NaN entries and formatting the test data



# Impute the missing values with the median

test.Fare[152] = test['Fare'].median()

test['Age'] = test['Age'].fillna(test['Age'].median())



# Convert gender to binary

# Male: 0

# Female: 1

test['Sex'][test['Sex'] == 'male'] = 0

test['Sex'][test['Sex'] == 'female'] = 1



# Convert the Embarked classes to integer form

test['Embarked'][test['Embarked'] == "S"] = 0

test['Embarked'][test['Embarked'] == "C"] = 1

test['Embarked'][test['Embarked'] == "Q"] = 2
#  Creating a target variable

target = train['Survived'].values



# Extracting useful features from the training set

train_features = train[['Pclass', 'Sex', 'Age', 'Fare']].values



# Extract the useful features from the test set

test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values
# Fit the first decision tree

decision_tree = DecisionTreeClassifier()

decision_tree = decision_tree.fit(train_features, target)
# Importance and the score of the included features

print('Feature Importance: ', decision_tree.feature_importances_)

print('Score:', decision_tree.score(train_features, target))
# Predict the survival rate

prediction = decision_tree.predict(test_features)
# Create a data frame with two columns: PassengerId & Survived.

# Survived contains the predictions

PassengerId = np.array(test['PassengerId']).astype(int)

survived = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])
survived.head()
survived.shape
# Write the solution to a csv file

survived.to_csv("survived.csv", index_label = ["PassengerId"])
train['family_size'] = train['Parch'] + train['SibSp'] + 1

test['family_size'] = test['Parch'] + test['SibSp'] + 1

train.head()
# Create a new feature array

features_family = train[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family_size']].values

test_features_family = test[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'family_size']].values
decision_tree_family = DecisionTreeClassifier()

decision_tree_family = decision_tree_family.fit(features_family, target)
decision_tree_family.score(features_family, target)
predictions_family = decision_tree_family.predict(test_features_family)
# Create a data frame with two columns: PassengerId & Survived.

# Survived contains the predictions

PassengerId = np.array(test['PassengerId']).astype(int)

solution_family = pd.DataFrame(predictions_family, PassengerId, columns = ['Survived'])
solution_family.head()
# Write the solution to a csv file

solution_family.to_csv("solution_family.csv", index_label = ["PassengerId"])
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(

    max_depth=10, min_samples_split=2, n_estimators=100, random_state=1

)

my_forest = forest.fit(features_forest, target)
# Forest score

print(my_forest.score(features_forest, target))
# Compute predictions the test set features

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)
# print the `.feature_importances_` attribute

print(my_forest.feature_importances_)



# Compute and print the mean accuracy score

print(my_forest.score(features_forest, target))
# Create a data frame with two columns: PassengerId & Survived.

# Survived contains the predictions

PassengerId = np.array(test['PassengerId']).astype(int)

solution_tree = pd.DataFrame(pred_forest, PassengerId, columns = ['Survived'])
solution_tree.head()
# Write the solution to a csv file

solution_tree.to_csv("solution_tree.csv", index_label = ["PassengerId"])