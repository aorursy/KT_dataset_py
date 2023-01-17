# import libraries

import numpy as np # linear algebra and arraus

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the train dataset to create a DataFrame

train = pd.read_csv("../input/train.csv")

# Count missing values in train – missing 177 Age and 2 Embarked values in train (ignoring missing cabin)

train.isnull().sum()



# Cleaning and Formatting the train Data 

# Filling missing Age values with median

train["Age"] = train["Age"].fillna(train["Age"].median())



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")
# Convert the male and female groups and Embarked classes to dummy variables using OneHotEncoding method

# but first we have to drop columns that aren't helpful and our target variable

# and differentiate between categorical and numeric columns

candidate_train_predictors = train.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)

categorical_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]



dummy_encoded_train_predictors = pd.get_dummies(train_predictors)
# Create the target and features numpy arrays: target, features_one

y_target = train["Survived"].values

x_features_one = dummy_encoded_train_predictors.values

print(x_features_one)
## Split the data into training and validation sets - 75%/25%

x_train, x_validation, y_train, y_validation = train_test_split(x_features_one,y_target,test_size=.25,random_state=1)
# Fit first decision tree: tree_one

tree_one = tree.DecisionTreeClassifier()

tree_one = tree_one.fit(x_features_one, y_target)
# Look at the score of the included features

tree_one_accuracy = round(tree_one.score(x_features_one, y_target), 4)

print("Accuracy: %0.4f" % (tree_one_accuracy))

# Predict y_target given validation set

predictions = tree_one.predict(x_validation)



## Look at the confusion matrix ([TN,FN],[FP,TP])

confusion_matrix(y_validation,predictions)
## score of predicted output from tree_one model against ground truth

accuracy_score(y_validation, predictions)
# Overfitting and additional features

# Create the target and features numpy arrays: target, features_two

y_target = train["Survived"].values

x_features_two = dummy_encoded_train_predictors.values
# Fit second decision tree: tree_two

tree_two = tree.DecisionTreeClassifier()

tree_two = tree_two.fit(x_features_two, y_target)



# Look at the score of the included features - SAME AS BEFORE

tree_two_accuracy = round(tree_two.score(x_features_two, y_target), 4)

print("Accuracy: %0.4f" % (tree_two_accuracy))

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5

max_depth = 10

min_samples_split = 5

tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)

tree_two = tree_two.fit(x_features_two, y_target)



# Look at the score of the included features

tree_two_accuracy = round(tree_two.score(x_features_two, y_target), 4)

print("Accuracy: %0.4f" % (tree_two_accuracy))

# Feature Engineering

# add column to original pandas dataframe and then re-do one hot encoding

train['family_size'] = 1 + train['SibSp'] + train['Parch']



candidate_train_predictors = train.drop(['PassengerId','Survived','Name','Ticket','Cabin','SibSp','Parch'], axis=1)

categorical_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]



dummy_encoded_train_predictors = pd.get_dummies(train_predictors)



# Create the target and features numpy arrays: target, features_three

y_target = train["Survived"].values

x_features_three = dummy_encoded_train_predictors.values
# Fit third decision tree: tree_three

tree_three = tree.DecisionTreeClassifier()

tree_three = tree_three.fit(x_features_three, y_target)
# Look at the score of the included features

tree_three_accuracy = round(tree_three.score(x_features_three, y_target), 4)

print("Accuracy: %0.4f" % (tree_three_accuracy))
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5

max_depth = 10

min_samples_split = 5

tree_three = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)

tree_three = tree_three.fit(x_features_three, y_target)
# Look at overfit controlled tree with Engineered Feature 'family_size'

# Look at the score of the included features

print(tree_three.score(x_features_three, y_target))