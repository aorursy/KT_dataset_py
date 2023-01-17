import numpy as np

import pandas as pd

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

combine=[train,test]
print((train["Survived"]==1))
print(train.columns.values)
train.head()
train.tail()
train.describe()
train.info()
#To see how many people survived

train["Survived"].value_counts()

#percentages

train["Survived"].value_counts(normalize=True)
#To see the relation with sex

#No of males that survived 

train["Survived"][train["Sex"]=='male'].value_counts()
train["Survived"][train["Sex"]=='female'].value_counts()
# Does age play a role

#Male who survived whose age were lesser than 50 yrs

train["Survived"][train["Age"]<50][train["Sex"]=='male'].value_counts()



#Creating a decision tree

print(train)
test_one=test

#test_one["Survived"]=0

train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]
train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]

train.shape, test.shape
#Cleaning and formatting the data



train["Age"]=train["Age"].fillna(train["Age"].median())

print(train)
#Sex and embarked do not contain numerical values

#male->0

# female->1







# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"]=="female"]=1



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



#Print the Sex and Embarked columns

print(train["Sex"])

print(train["Embarked"])

print(train)


#train["Age"]=train["Age"].fillna(train["Age"].median())

#print(train["Age"])



test.Fare[152]=test.Fare.median()

from sklearn import tree

target=train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

clf=tree.DecisionTreeClassifier()

clf=clf.fit(features_one,target)

print(clf.feature_importances_)

print(clf.score(features_one, target))

# Fare is the most important featue in prediction
#Preprocessing of test data

test.Fare[152]=test.Fare.median()

test["Age"]=test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"]=="female"]=1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2







print(test)


test_features=test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction=clf.predict(test_features)

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)

print(my_solution.shape)

my_solution.to_csv("C:\\Users\\Ayushi Asthana\\Documents\\Machine learning\\Titanic\\gender_submission.csv", index_label = ["PassengerId"])

train["Fare"]=train["Fare"].fillna(train["Fare"].median())
print(train)
# To avoid Overfitting

# Create a new array with the added features: features_two

train["Age"]=train["Age"].fillna(train["Age"].median())

features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch","Embarked"]].values

print(features_two)





#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth =10 

min_samples_split =5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)



my_tree_two = my_tree_two.fit(features_two,target)



#Print the score of the new decison tree

print(my_tree_two.feature_importances_)

print(my_tree_two.score(features_two, target))



test_features=test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction=clf.predict(test_features)

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)

print(my_solution.shape)





print(test)
# Create train_two with the newly defined feature

train_two = train.copy()

train_two["family_size"] = train_two["SibSp"]+train_two["Parch"]+1



# Create a new feature set and add the new feature

features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch","family_size"]].values



# Define the tree classifier, then fit the model

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three,target)



# Print the score of this decision tree

print(my_tree_three.score(features_three, target))

my_solution.to_csv("C:\\Users\\Ayushi Asthana\\Documents\\Machine learning\\Titanic\\gender_submission2.csv", index_label = ["PassengerId"])

# Using Random Forest

from sklearn.ensemble import RandomForestClassifier

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest,target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))



# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred_forest

    })

submission.to_csv('gender_submissionrandom.csv', index=False)



#Request and print the `.feature_importances_` attribute

print(my_tree_two.feature_importances_)

print(my_forest.feature_importances_)





#Compute and print the mean accuracy score for both models

print(my_tree_two.score(features_two, target))

print(my_forest.score(features_forest,target))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
import xgboost as xgb

from xgboost import XGBClassifier
gbm=xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=1).fit(features_two,target)
predictions=gbm.predict(test_features)

submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submissionxgboost2.csv", index=False)

print(gbm.score(features_two,target))