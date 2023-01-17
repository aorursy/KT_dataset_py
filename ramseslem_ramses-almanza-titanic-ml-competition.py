import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

#Creating Dataframes with the csv files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

"""Sex is a categorical data so we need to deal with it
changing the female value to 1 and the male value to 0"""

train["Sex"][train["Sex"] == 'female'] = 1
train["Sex"][train["Sex"] == 'male'] = 0

"""Age column has Nan values so we have to deal with those values assigning them the median 
    value from the entire column"""

train["Age"] = train["Age"].fillna(train["Age"].median())
#Now we make our predictions on the training set with the following Features
#Import sklearn library for decision trees
from sklearn import tree
target = train["Survived"].values
train_features = train[["Pclass", "Sex", "Age", "Fare"]].values

prediction_tree = tree.DecisionTreeClassifier()
prediction_tree = prediction_tree.fit(train_features, target)
# Look at the importance and score of the included features
print(prediction_tree.feature_importances_)
print(prediction_tree.score(train_features, target))
#Now its time to test my tree into test set but first I have to look at my test set
#Sex variable is a categorical variable so I need to convert in a dummy variable
test["Sex"][test["Sex"] == 'female'] = 1
test["Sex"][test["Sex"] == 'male'] = 0
#Age variable in test set has NaN values so I need to fill those values with the median
test["Age"] = test["Age"].fillna(test["Age"].median())
#The row 152 from Fare column is a NaN value so I need to fill it
test.Fare[152] = test.Fare.median()
#Now its time to predict using thw test set
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
test_prediction = prediction_tree.predict(test_features)
print(test_prediction)
#Dealing with NaN values from Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


#Adding new features
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

prediction_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
prediction_tree_two = prediction_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(prediction_tree_two.score(features_two, target))
#Adding Feature Engineering to the equation
train_two = train.copy()
train_two["family_size"] = train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
prediction_tree_three = tree.DecisionTreeClassifier()
prediction_tree_three = prediction_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(prediction_tree_three.score(features_three, target))
#Working with test set
#Dealing with NaN values from Embarked variable
test["Embarked"] = test["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test_two = test.copy()
test_two["family_size"] = test["SibSp"] + test["Parch"] + 1

# Create a new feature set and add the new feature
features_three_test = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

test_prediction_two = prediction_tree_three.predict(features_three_test)
print(test_prediction_two)
# Create a data frame with two columns: PassengerId & Survived. Survived contains my predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_three = pd.DataFrame(test_prediction_two, PassengerId, columns = ["Survived"])
print(my_solution_three)
my_solution_three.to_csv("my_solution_three.csv", index_label = ["PassengerId"])
#I will use Random Forest to improve my predictions
from sklearn.ensemble import RandomForestClassifier

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
print(my_forest.score(features_forest, target))
#Predict results to my test set
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))
#Another round of Forest
my_forest_two = forest.fit(features_three, target)
print(my_forest_two.score(features_three, target))
#made no improvemsts on the second round of random forest
