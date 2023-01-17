import pandas as pd
import numpy as np # For linear algebra
import matplotlib.pyplot as plt # For visualization
from sklearn import tree # For creating trees

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Fun fact: We can instantly create a classifier with a 0.76555 public score with the 2 lines of code below.
# It is analogous to gender_submission.csv:
# test["Survived"] = 0
# test["Survived"][test["Sex"] == "female"] = 1
train.head()
train.drop("Ticket", 1, inplace = True)
test.drop("Ticket", 1, inplace = True)
train.shape
train.describe()
train["Survived"].value_counts()
train["Survived"][train["Sex"] == "female"].value_counts(normalize = True) # normalize = True returns percentages instead of raw counts
train.isnull().sum()
test.isnull().sum()
fig, ax = plt.subplots()
ax.hist(train["Embarked"])
ax.set_title("Embarked Classes")
plt.show()
train["Embarked"].fillna("S", inplace = True)
median_age_train = train["Age"].dropna().median()
median_age_test = test["Age"].dropna().median()
print(median_age_train, median_age_test)
train["Age"].fillna(median_age_train, inplace = True)
test["Age"].fillna(median_age_train, inplace = True)
test["Fare"].fillna(test["Fare"].mean(), inplace = True) # It would have been better of course to use the mean for his/her passenger class.
def getCabinCat(cabin_code):
    if pd.isnull(cabin_code):
        cat = "Unknown"
    else:
        cat = cabin_code[0]
    return cat

cabin_cats_train = np.array([getCabinCat(cc) for cc in train["Cabin"].values])
cabin_cats_test = np.array([getCabinCat(cc) for cc in test["Cabin"].values])

# We can now add this as a new "CabinCat" feature in the DataFrames, and remove the "Cabin" feature.
train = train.assign(CabinCat = cabin_cats_train)
train = train.drop("Cabin", axis = 1)
test = test.assign(CabinCat = cabin_cats_test)
test = test.drop("Cabin", axis = 1)

# We can now investigate the distribution of passengers over the cabin categories in the training set:
print("Number of passengers:\n{}".format(train["CabinCat"].groupby(train["CabinCat"]).size()))
train.isnull().sum()
test.isnull().sum()
train["Child"] = np.where(train["Age"] < 18, 1, 0)
# NB all passengers with missing ages were imputed an age which was > 18, which is not ideal since some of them could have been children.
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
from sklearn.preprocessing import LabelEncoder
categorical_classes_list = ["Sex", "Embarked"]
#encode features that are cateorical classes
for column in categorical_classes_list:
    le = LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    test[column] = le.transform(test[column])
# Create the features and target numpy arrays: features_one, target
features_one = train[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values
target = train["Survived"].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
import graphviz # Open source graph visualization software
dot_data = tree.export_graphviz(my_tree_one, out_file = None) # Instead of outputing a .out file with the description of the model, we store it in a variable.
graph = graphviz.Source(dot_data) 
graph
my_prediction = my_tree_one.predict(test[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
# Create a new array of features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1) # For a good explanation of what the random_state does, see here: https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))
# Create train_two with the newly defined feature
train_two = train.copy()
train_two["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "FamilySize"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree, and remember lower is probably better
print(my_tree_three.score(features_three, target))
# Import the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a features variable with the features we want
features_forest = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

print(my_forest.score(features_forest, target))
# Compute predictions on our test set features
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)

# Create DataFrame and .csv file
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution_random_forest.csv", index_label = ["PassengerId"])
# The files can be found here:
import os
print(os.listdir("../working"))