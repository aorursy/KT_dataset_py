# Import the Numpy library
import numpy as np
import pandas as pd 
# Import 'tree' from scikit-learn library
from sklearn import tree
import seaborn as sb
from matplotlib import pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Load the train and test datasets to create two DataFrames
train_url = '../input/train.csv'
train = pd.read_csv(train_url)

test_url = '../input/test.csv'
test=pd.read_csv(test_url)
#Print the `head` of the train dataframes
train.head()
#Print the `head` of the test dataframes
test.head()
train.describe()
test.describe()
print(train.keys())
print(test.keys())
# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())
# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))
# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] >= 18]=0
train["Child"][train["Age"] < 18]=1
# print(train['Child'])
# Create a copy of test: test_one
test_one=test.copy()

# Initialize a Survived column to 0
test_one['Survived']=0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one['Sex']=='female']=1
print(test_one['Survived'])
train["Sex"].head()
# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Convert the male and female groups to integer form
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
train["Embarked"].head()
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
# test dataset
# Impute the Embarked variable
test["Embarked"] = test["Embarked"].fillna('S')

# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
#Print the Sex columns
print(train['Sex'].head())
 #Print the Embarked column
print(train['Embarked'].head())
def null_table(training, testing):
    print("Training Data Frame")
    print(pd.isnull(training).sum()) 
    print(" ")
    print("Testing Data Frame")
    print(pd.isnull(testing).sum())

null_table(train, test)


train.drop(labels = ['Cabin', 'Ticket','Child'], axis = 1, inplace = True)
test.drop(labels = ['Cabin', 'Ticket'], axis = 1, inplace = True)

null_table(train, test)
#the median will be an acceptable value to place in the NaN cells
train['Age'].fillna(train['Age'].median(), inplace = True)
test["Age"].fillna(test["Age"].median(), inplace = True) 
train["Embarked"].fillna("S", inplace = True)
test["Fare"].fillna(test["Fare"].median(), inplace = True)

null_table(train, test)


sb.barplot(x="Sex", y="Survived", data=train)
plt.title("Distribution of Survival based on Gender")
plt.show()

total_survived_females = train[train.Sex == 1]["Survived"].sum()
total_survived_males = train[train.Sex == 0]["Survived"].sum()

print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))
sb.barplot(x="Pclass", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival Based on Class")
plt.show()
sb.barplot(x="Pclass", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")
plt.show()
sb.barplot(x="Sex", y="Survived", hue="Pclass", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# Print the train data to see the available features
# print(train)
# Create the target and features numpy arrays: target, features_one
target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
# Impute the missing value with the median
# test.Fare[152] = 

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split =5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two,target))
# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"]+train_two["Parch"]+1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest,target))
# Print the score of this decision tree
print(my_tree_three.score(features_three, target))
print(my_forest.score(features_forest,target))
# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest2 = features_three
test_forest2 = test.copy()
test_forest2["family_size"] = test_forest2["SibSp"]+test_forest2["Parch"]+1
# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest2, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest2, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test_forest2[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "family_size"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
# print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])