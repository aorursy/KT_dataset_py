# Load packages

import numpy as np  
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import re
# Load training data

train_set = pd.read_csv("../input/train.csv")
test_set  = pd.read_csv("../input/test.csv")
# Review input features - Part 1

print("Shape of training set:", train_set.shape, "\n")
print("Column Headers:", list(train_set.columns.values), "\n")
print("Shape of test set:", test_set.shape, "\n")
print(train_set.describe())

# preview the data
train_set.head()
# Review input features (train set) - Part 2A

missing_values = []
nonumeric_values = []

print ("TRAINING SET INFORMATION")
print ("========================\n")
for column in train_set:
    # Find all the unique feature values
    uniq = train_set[column].unique()
    print ("'{}' has {} unique values" .format(column,uniq.size))
    if (uniq.size > 25):
        print("~~Listing up to 25 unique values~~")
    print (uniq[0:24])
    print ("\n-----------------------------------------------------------------------\n")
    
    # Find features with missing values
    if (True in pd.isnull(uniq)):
        s = "{} has {} missing" .format(column, pd.isnull(train_set[column]).sum())
        missing_values.append(s)
    
    # Find features with non-numeric values
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            break
        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):
            nonumeric_values.append(column)
            break
  
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print ("Features with missing values:\n{}\n\n" .format(missing_values))
print ("Features with non-numeric values:\n{}" .format(nonumeric_values))
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
# Review input features (test set) - Part 2B

missing_values = []
nonumeric_values = []

print ("TEST SET INFORMATION")
print ("====================\n")

for column in test_set:
    # Find all the unique feature values
    uniq = test_set[column].unique()
    print ("'{}' has {} unique values" .format(column,uniq.size))
    if (uniq.size > 25):
        print("~~Listing up to 25 unique values~~")
    print (uniq[0:24])
    print ("\n-----------------------------------------------------------------------\n")
    
    # Find features with missing values
    if (True in pd.isnull(uniq)):
        s = "{} has {} missing" .format(column, pd.isnull(test_set[column]).sum())
        missing_values.append(s)
    
    # Find features with non-numeric values
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            break
        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):
            nonumeric_values.append(column)
            break
  
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print ("Features with missing values:\n{}\n\n" .format(missing_values))
print ("Features with non-numeric values:\n{}" .format(nonumeric_values))
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
# Feature Cleaning

# Convert non-numeric values for Sex, Embarked
# male=0, female=1
train_set.loc[train_set["Sex"] == "male", "Sex"]   = 0
train_set.loc[train_set["Sex"] == "female", "Sex"] = 1

test_set.loc[test_set["Sex"] == "male", "Sex"]   = 0
test_set.loc[test_set["Sex"] == "female", "Sex"] = 1

# S=0, C=1, Q=2
train_set.loc[train_set["Embarked"] == "S", "Embarked"] = 0
train_set.loc[train_set["Embarked"] == "C", "Embarked"] = 1
train_set.loc[train_set["Embarked"] == "Q", "Embarked"] = 2

test_set.loc[test_set["Embarked"] == "S", "Embarked"] = 0
test_set.loc[test_set["Embarked"] == "C", "Embarked"] = 1
test_set.loc[test_set["Embarked"] == "Q", "Embarked"] = 2

# Substitute missing values for Age, Embarked & Fare
train_set["Age"]      = train_set["Age"].fillna(train_set["Age"].median())
train_set["Fare"]     = train_set["Fare"].fillna(train_set["Fare"].median())
train_set["Embarked"] = train_set["Embarked"].fillna(train_set["Embarked"].median())

test_set["Age"] = test_set["Age"].fillna(test_set["Age"].median())
test_set["Fare"] = test_set["Fare"].fillna(test_set["Fare"].median())

print ("Converted non-numeric features for Sex & Embarked...\nSubstituted missing values for Age, Embarked & Fare")
# Visualize the features
plt.style.use('ggplot')

selected_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
train_set[selected_columns].hist()
# Features used for training
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Train / Test split for original training data
# Withold 20% from train set for testing
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_set[predictors], train_set["Survived"], test_size=0.2, random_state=0)

print ("Original Training Set: {}\nTraining Set: {}\nTesting Set(witheld): {}" .format(train_set.shape, X_train.shape,X_test.shape))


# Normalize features - both training & test (withheld & final)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
final_test_transformed  = scaler.transform(test_set[predictors])

print ("Transformed training, test sets (withheld & final)")

# Scoring Metric - Accuracy
print ("Use accuracy as the score function")
# Use a simple model
# Import the logistic regression class
from sklearn.linear_model import LogisticRegression

# Initialize the algorithm
# Logistic regression defaults to mean accuracy as score
alg = LogisticRegression(random_state=1)

clf = alg.fit(X_train_transformed, y_train)

# Scores
train_score = clf.score(X_train_transformed, y_train)
test_score  = clf.score(X_test_transformed, y_test)

print ("Train Score: {}\nTest Score: {}" .format(train_score, test_score))
# Use Cross Validation

# Compute accuracy for all the cross validation folds
scores = cross_validation.cross_val_score(alg, X_train_transformed, y_train, cv=3, scoring='accuracy')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Run Diagnostics

# Perform feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)
selector.fit(X_train, y_train)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
# Attempt LogisticRegressionCV which performs hyper parameter tuning

# Subset of Features used for training
#predictors = ["Pclass", "Sex", "Parch", "Fare", "Embarked"]
#predictors = ["Pclass", "Sex", "Fare"]
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# Train / Test split for original training data
# Withold 20% from train set for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train_set[predictors], train_set["Survived"], test_size=0.2, random_state=0)

print ("Original Training Set: {}\nTraining Set: {}\nTesting Set(witheld): {}" .format(train_set.shape, X_train.shape,X_test.shape))


# Normalize features - both training & test (withheld & final)
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
final_test_transformed  = scaler.transform(test_set[predictors])

print ("Transformed training, test sets (withheld & final)")

# Scoring Metric - Accuracy
print ("Use accuracy as the score function")
# Use a simple model
# Import the logistic regression class
from sklearn.linear_model import LogisticRegressionCV

# Initialize the algorithm
# Logistic regression defaults to mean accuracy as score
alg = LogisticRegressionCV(random_state=1)

# Fit to training data
clf = alg.fit(X_train_transformed, y_train)
print (clf.get_params())

# Scores
train_score = clf.score(X_train_transformed, y_train)
test_score  = clf.score(X_test_transformed, y_test)
print ("\nTrain Score: {}\nTest Score: {}" .format(train_score, test_score))
# Make Predictions using Test Set

# Make predictions using the test set.
predictions = clf.predict(final_test_transformed)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic_lr2.csv', index=False)

submission.head(15)
# Implement Grid Search
from sklearn.grid_search import GridSearchCV
alg2 = LogisticRegressionCV(random_state=1)
tuned_parameters = [{'solver': ['liblinear', 'lbfgs']}]

clf2 = GridSearchCV(alg2, tuned_parameters, cv=3, scoring='accuracy')
clf2.fit(X_train, y_train)
print(clf2.best_params_)
print(clf2.get_params())

# Scores
train_score = clf2.score(X_train_transformed, y_train)
test_score  = clf2.score(X_test_transformed, y_test)

print ("Train Score: {}\nTest Score: {}" .format(train_score, test_score))
# Run Diagnostics - LogisticRegressionCV
predictions_train = alg.predict(train_set[predictors])

print (type(predictions_train))
predictions_train[1]
print (type(train_set))
train_set.loc[1, "Survived"]

miss = []

for i in range (1, predictions_train.size):
    if (predictions_train[i] != train_set.loc[i, "Survived"]):
        miss.append(i)
missclass = train_set.loc[miss, ['Survived','Sex','Pclass','Parch','Fare','Embarked']]
missclass.hist()
print (missclass.describe())
# Make Predictions using Test Set

# Make predictions using the test set.
predictions = alg.predict(final_test_transformed)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic.csv', index=False)
submission.head()