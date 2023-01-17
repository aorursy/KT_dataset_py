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

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

full_data = [titanic_df, test_df]


# preview the data
titanic_df.head(10)

titanic_df.tail(10)

titanic_df.shape
#overview of our variable

print(titanic_df.columns.values)

y_titanic_df=titanic_df.iloc[:,1]

#percentages 

titanic_df["Survived"].value_counts(normalize = True)
#anotherway ??? 

print (titanic_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
#male survival

print(titanic_df["Survived"][titanic_df["Sex"] == "male"].value_counts(normalize = True))
#female survial

print(titanic_df['Survived'][titanic_df['Sex'] == 'female'].value_counts(normalize = True))
sns.barplot(titanic_df["Sex"],y_titanic_df)

# Create the column Child 
titanic_df["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older
titanic_df["Child"][titanic_df["Age"] < 18] = 1
titanic_df["Child"][titanic_df["Age"] >= 18] = 0
# normalized Survival Rates for passengers under 18
print(titanic_df["Survived"][titanic_df["Child"] == 1].value_counts(normalize = True)) 

# normalized Survival Rates for passengers 18 or older

print(titanic_df["Survived"][titanic_df["Child"] == 0].value_counts(normalize = True))

#Checking for missing values:

titanic_df.isnull().sum() #Missing values in train data
test_df.isnull().sum() #Mising values in test data

titanic_df.Age.isnull().any() #So no null values left finally 

#take care of embarked

titanic_df['Embarked'].fillna('S',inplace=True)
titanic_df.Embarked.isnull().any()

# Finally No NaN values
#exercing for loop

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(titanic_df['Fare'].median())
titanic_df['CategoricalFare'] = pd.qcut(titanic_df['Fare'], 4)
print (titanic_df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
#split survivor by age

for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
titanic_df['CategoricalAge'] = pd.cut(titanic_df['Age'], 5)

print (titanic_df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
#dummy variable

titanic_df["Sex"][titanic_df["Sex"] == "male"] = 0
titanic_df["Sex"][titanic_df["Sex"] == "female"] = 1
from sklearn.preprocessing import LabelEncoder 

#remove cabin 

titanic_df.drop('Cabin',axis = 1, inplace = True)

# Create a copy of test: test_one

test_one = test_df

# Initialize a Survived column to 0

test_one["Survived"] = 0 

# Survived = 1 if Sex equals "female"

test_one["Survived"][test_one["Sex"] == "female"] = 1

print(test_one.Survived)
# library for machine learning 

from sklearn import tree
#overview of variable needed 

print(titanic_df)

# Create the target and features numpy arrays: target, features_one

target = titanic_df["Survived"].values

features_one = titanic_df[["Pclass", "Sex", "Age", "Fare"]].values
# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

test_df.isnull().sum() #Mising values in test data

test_df.drop('Cabin',axis = 1, inplace = True)

# Impute the missing value with the median

test_df.Fare[152] = test_df.Fare.median()



for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (titanic_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# Create a new feature set and add the new feature
features_three = titanic_df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values
# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)
# Print the score of this decision tree
mprint(my_tree_three.score(features_three, target))
print(titanic_df["Sex"])
# Impute the Embarked variable
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
# Convert the Embarked classes to integer form
titanic_df["Embarked"][titanic_df["Embarked"] == "S"] = 0
titanic_df["Embarked"][titanic_df["Embarked"] == "C"] = 1
titanic_df["Embarked"][titanic_df["Embarked"] == "Q"] = 2 # Print the Sex and Embarked columns
print(titanic_df["Sex"])

# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier 
features_forest = titanic_df[["Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values

# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test_df[["Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))
print(my_forest.feature_importances_)

importances = my_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)

plt.figure()
plt.title("Feature importances")
plt.barh(range(features_forest.shape[1]), importances[indices],
       color="r", xerr=std[indices], align="center")
plt.yticks(range(features_forest.shape[1]), indices)
plt.ylim([-1, features_forest.shape[1]])
plt.show()
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred_forest
    })
submission.head()
submission.to_csv('submission.csv', index= False)
