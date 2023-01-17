import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.dtypes)
print(train.head())
print(train.describe())
print(train.mode())

### Data exploration
# Predictor variables:
# PassengerId - Int - Continuos - Does not have much significance as its unique field. Excluded.
# Pclass  - Int - Categorical - socio economic class - 1,2,3 - Included
# Name - string - char - Excluded for now. There could be racial angle to survival and we could derive it by name but it will take a lot of work.
# Sex - char - Categorical - male, female - Included
# Age  - Int - continuos - Included.
# SibSp  - int - continuos - number of siblings, 0-6. Included. Please with family members to have low survival rate due to dependency.
# Parch - int - continuos - number of parents - 0-6. Included. Please with family members to have low survival rate due to dependency.
# Ticket - string - ticket number -Excluded for now. multiple could have same ticket numbe depending on how they are bought    
# Fare - float - price of the ticket. Included. 
# Cabin - string - cabin number - contains 687 NANs. Excluded.
# Embarked  - S, C, Q. 2 NAN so needs clean up. Replace with Mode which "S".Included.
# Child  - Calculated column based on age. Included
# family_size   - Calculated field. Self + Parch + SibSp. We might just include this one field instead of 3.

# Target variable - 
# Survived - Int - Continuos

# Histograms
#Pclass
print(train["Pclass"].value_counts(dropna=False))
plt.hist(train["Pclass"])
plt.title("Pclass Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#Age
#print(train["Age"].value_counts(dropna=False))
plt.hist(train["Age"].dropna(), bins=8)
plt.title("Age Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

### Data imputaions and cleanup

# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')
test["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train.loc[train["Age"] < 18, "Child"] = 1
train.loc[train["Age"] >= 18, "Child"] = 0
train["Age"] = train["Age"].fillna(-1)

# Age imputation
test.loc[test["Age"] < 18, "Child"] = 1
test.loc[test["Age"] >= 18, "Child"] = 0
#test["Age"] = test["Age"].fillna(-1)



#Convert the male and female groups to integer form
train.loc[train.Sex == "male", "Sex"] = 0
train.loc[train.Sex == "female", "Sex"] = 1

test.loc[test.Sex == "male", "Sex"] = 0
test.loc[test.Sex == "female", "Sex"] = 1

#Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# Impute the missing value with the median
test.loc[152, "Fare"] = np.nanmedian(test["Fare"])

# Create train_two with the newly defined feature
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1

#Print the `head` of the train and test dataframes
print(train.head())
print(train.describe())
#print(test.head())
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# remove women with cabin and men without cabin
# Women with cabin survive 93% of the times
# Men without cabin dont survive 87% of the times
#train_new = train[(~((train["Sex"] == 0) & (pd.isnull(train["Cabin"]) == False))) & (~((train.Sex == 1) & (pd.isnull(train.Cabin) == True)))]
train_new = train.copy()
#print(train_new.shape)

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train_new[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
target = train_new[["Survived"]].values


# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))
print(my_forest.feature_importances_)

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# pre assignment print
#print(test["SurvivedPredicted"][(((test["Sex"] == 0) & (pd.isnull(test["Cabin"]) == False)))].value_counts())
#print(test["SurvivedPredicted"][(((test.Sex == 1) & (pd.isnull(test.Cabin) == True)))].value_counts())
#& (~((test.Sex == 1) & (pd.isnull(test.Cabin) == True)))]

# Take a default shot that women with cabin have all survived and men without cabin have all perished
#test["SurvivedPredicted"][(((test["Sex"] == 0) & (pd.isnull(test["Cabin"]) == False)))] = 1
#test["SurvivedPredicted"][(((test.Sex == 1) & (pd.isnull(test.Cabin) == True)))] = 0

# post assignment print
#print(test["SurvivedPredicted"][(((test["Sex"] == 0) & (pd.isnull(test["Cabin"]) == False)))].value_counts())
#print(test["SurvivedPredicted"][(((test.Sex == 1) & (pd.isnull(test.Cabin) == True)))].value_counts())

#pred_forest = test["SurvivedPredicted"].values

#print(pred_forest)
#print(len(pred_forest)