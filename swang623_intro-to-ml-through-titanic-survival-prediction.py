# Data Analysis Libraries

import numpy as np

import pandas as pd



# Data Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Machine Learning Libraries

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# A basic look at the training data

train.sample(5)
# Summary of the training data

train.describe(include = "all")
# Get a clearer understanding of data types and missing values

train.info()

print('***************************************************')

test.info()
# Explore if survival rate depends on passenger class

sns.barplot(x = "Pclass", y = "Survived", data = train)

train[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean()
# Explore if survival rate depends on passenger gender

sns.barplot(x = "Sex", y = "Survived", data = train)

train[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean()
# Age is a continuous variable with 20% of the data missing. 

# We will first look at the distribution

sns.distplot(train["Age"].dropna(), bins = 70, kde = False)
# Explore if survival rate depends on the number of siblings/spouses abroad the Titanic

sns.barplot(x = "SibSp", y = "Survived", data = train)

sibsp = pd.DataFrame()

sibsp["Survived Mean"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean()["Survived"]

sibsp["Count"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).count()["Survived"]

sibsp["STD"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).std()["Survived"]

print(sibsp)

train[(train["SibSp"] == 5)|(train["SibSp"] == 8)]
# Explore if survival rate depends on the number of parents/children abroad the Titanic

sns.barplot(x = "Parch", y = "Survived", data = train)

sibsp["Survived Mean"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean()["Survived"]

sibsp["Count"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).count()["Survived"]

sibsp["STD"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).std()["Survived"]

print(sibsp)
# See the distribution of Fare

#sns.distplot(train["Fare"][train["Pclass"]==1].dropna(), bins = 10, kde = False)

print(train[["Fare", "Survived"]].dropna().groupby(["Survived"]).count())

fare_hist = sns.FacetGrid(train, col="Survived")

fare_hist = fare_hist.map(plt.hist, "Fare")



train[["Fare", "Survived"]].dropna().groupby(["Survived"]).median()
# There are many missing values in this colomn

(train["Survived"][train["Cabin"].isnull()].count())/(train["Cabin"].count())
# Explore if survival rate depends on the port passenger embarked

sns.barplot(x = "Embarked", y = "Survived", data = train)

train[["Survived", "Embarked"]].groupby(["Embarked"]).mean()
PassengerId = test['PassengerId']

train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)

test = test.drop(["PassengerId","Name", "Ticket", "Cabin"], axis = 1)
#train.info()

#print('***************************************************')

#test.info()
print(train["Embarked"].unique())

print(train.groupby(["Embarked"])["Survived"].count().reset_index())

train["Embarked"] = train["Embarked"].fillna("S")

train.groupby(["Embarked"])["Survived"].count().reset_index()
# Calculate mean and standard deviation of "Age" column

train_mean = train["Age"].mean()

train_std = train["Age"].std()

test_mean = test["Age"].mean()

test_std = test["Age"].std()



# Count missing values

count_na_train = train["Age"].isnull().sum()

count_na_test = test["Age"].isnull().sum()



# generate random numbers

np.random.seed(66)

train_rand = np.random.randint(train_mean - train_std, train_mean + train_std, size = count_na_train)

test_rand = np.random.randint(test_mean - test_std, test_mean + test_std, size = count_na_test)



# Fill missing values with random numbers

train["Age"][np.isnan(train["Age"])] = train_rand

test["Age"][np.isnan(test["Age"])] = test_rand



# Convert into int

train["Age"] = train["Age"].astype(int)

test["Age"] = test["Age"].astype(int)

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
# Confirm that all missing values are taken care of

#train.info()

#print('***************************************************')

#test.info()
# Map Age to categorical groups

train.loc[train["Age"] <= 16, "age_c"] = "1"

train.loc[(train["Age"] <= 32)&(train["Age"] > 16), "age_c"] = "2"

train.loc[(train["Age"] > 32)&(train["Age"] <= 48), "age_c"] = "3"

train.loc[(train["Age"] > 48)&(train["Age"] <= 64), "age_c"] = "4"

train.loc[(train["Age"] > 64), "age_c"] = "5"



test.loc[test["Age"] <= 16, "age_c"] = "1"

test.loc[(test["Age"] <= 32)&(test["Age"] > 16), "age_c"] = "2"

test.loc[(test["Age"] > 32)&(test["Age"] <= 48), "age_c"] = "3"

test.loc[(test["Age"] > 48)&(test["Age"] <= 64), "age_c"] = "4"

test.loc[(test["Age"] > 64), "age_c"] = "5"



train[["age_c","Survived"]].groupby(["age_c"]).mean()
# set up two new dataframes for the final model

m_train = train

m_test = test



#m_train.info()

#print('***************************************************')

#m_test.info()
# Generate Dummy Variable for Age

# Dropped the first one to avoid multicollinearity

#age_dummy_train = pd.get_dummies(train["age_c"], drop_first = True)

#age_dummy_test = pd.get_dummies(test["age_c"], drop_first = True)



# Concatenate Age dummy with the original training dataset

#m_train = pd.concat([m_train, age_dummy_train], axis = 1)

#m_test = pd.concat([m_test, age_dummy_test], axis = 1)



# Drop original Age and age_c

#m_train = m_train.drop(["age_c", "Age"], axis = 1)

#m_test = m_test.drop(["age_c", "Age"], axis = 1)



#m_train.sample(5)
# Map SibSp into categories

train.loc[train["SibSp"] == 0, "sib_c"] = "0"

train.loc[train["SibSp"] == 1, "sib_c"] = "1"

train.loc[train["SibSp"] >1 , "sib_c"] = "2"



test.loc[test["SibSp"] == 0, "sib_c"] = "0"

test.loc[test["SibSp"] == 1, "sib_c"] = "1"

test.loc[test["SibSp"] >1 , "sib_c"] = "2"



# Generate Dummy Variable

#sib_dummy_train = pd.get_dummies(train["sib_c"], drop_first = True)

#sib_dummy_test = pd.get_dummies(test["sib_c"], drop_first = True)





# Append sib_dummy to m-train

#m_train = pd.concat([m_train, sib_dummy_train], axis = 1)

#m_train = m_train.drop(["SibSp"], axis = 1)



#m_test = pd.concat([m_test, sib_dummy_test], axis = 1)

#m_test = m_test.drop(["SibSp"], axis = 1)



#m_train.sample(5)
# Map Parch into categories

train.loc[train["Parch"] == 0, "pc_c"] = "0"

train.loc[train["Parch"] == 1, "pc_c"] = "1"

train.loc[train["Parch"] >1 , "pc_c"] = "2"



test.loc[test["Parch"] == 0, "pc_c"] = "0"

test.loc[test["Parch"] == 1, "pc_c"] = "1"

test.loc[test["Parch"] >1 , "pc_c"] = "2"



# Generate Dummy Variable

#pc_dummy_train = pd.get_dummies(train["pc_c"], drop_first = True)

#pc_dummy_test = pd.get_dummies(test["pc_c"], drop_first = True)





# Append sib_dummy to m-train/m-test

#m_train = pd.concat([m_train, pc_dummy_train], axis = 1)

#m_train = m_train.drop(["Parch"], axis = 1)



#m_test = pd.concat([m_test, pc_dummy_test], axis = 1)

#m_test = m_test.drop(["Parch"], axis = 1)



#m_train.sample(5)
# Map fare values into categories

train["fare_c"] = pd.qcut(train["Fare"], 4, labels = ["1", "2", "3","4"])

test["fare_c"] = pd.qcut(test["Fare"], 4, labels = ["1", "2", "3","4"])



# Generate dummy variables for both train and test

#fare_dummy_train = pd.get_dummies(train["fare_c"], drop_first = True)

#fare_dummy_test = pd.get_dummies(test["fare_c"], drop_first = True)



# Append dummy variables to the original data frames

#m_train = pd.concat([m_train, fare_dummy_train], axis = 1)

#m_train = m_train.drop(["Fare"], axis = 1)



#m_test = pd.concat([m_test, fare_dummy_test], axis = 1)

#m_test = m_test.drop(["Fare"], axis = 1)



#m_train.sample(5)
train.loc[train["Sex"] == "male", "sex_c"] = "0"

train.loc[train["Sex"] == "female", "sex_c"] = "1"



test.loc[test["Sex"] == "male", "sex_c"] = "0"

test.loc[test["Sex"] == "female", "sex_c"] = "1"



# Generate dummy variables for both train and test

#sex_dummy_train = pd.get_dummies(train["Sex"], drop_first = True)

#sex_dummy_test = pd.get_dummies(test["Sex"], drop_first = True)



# Append dummy variables to the original data frames

#m_train = pd.concat([m_train, sex_dummy_train], axis = 1)

#m_train = m_train.drop(["Sex"], axis = 1)



#m_test = pd.concat([m_test, sex_dummy_test], axis = 1)

#m_test = m_test.drop(["Sex"], axis = 1)



#m_train.sample(5)
train.loc[train["Embarked"] == "S", "emk_c"] = "0"

train.loc[train["Embarked"] == "Q", "emk_c"] = "1"

train.loc[train["Embarked"] == "C", "emk_c"] = "2"



test.loc[test["Embarked"] == "S", "emk_c"] = "0"

test.loc[test["Embarked"] == "Q", "emk_c"] = "1"

test.loc[test["Embarked"] == "C", "emk_c"] = "2"



# Generate dummy variables for both train and test

#emk_dummy_train = pd.get_dummies(train["Embarked"], drop_first = True)

#emk_dummy_test = pd.get_dummies(test["Embarked"], drop_first = True)



# Append dummy variables to the original data frames

#m_train = pd.concat([m_train, emk_dummy_train], axis = 1)

#m_train = m_train.drop(["Embarked"], axis = 1)



#m_test = pd.concat([m_test, emk_dummy_test], axis = 1)

#m_test = m_test.drop(["Embarked"], axis = 1)



#m_train.sample(5)



train.sample(5)
# Map Parch into categories

#train.loc[train["Pclass"] == 1, "class_c"] = "class1"

#train.loc[train["Pclass"] == 2, "class_c"] = "class2"

#train.loc[train["Pclass"] == 3, "class_c"] = "class3"



#test.loc[train["Pclass"] == 1, "class_c"] = "class1"

#test.loc[train["Pclass"] == 2, "class_c"] = "class2"

#test.loc[train["Pclass"] == 3, "class_c"] = "class3"



# Generate dummy variables for both train and test

#class_dummy_train = pd.get_dummies(train["class_c"], drop_first = True)

#class_dummy_test = pd.get_dummies(test["class_c"], drop_first = True)



# Append dummy variables to the original data frames

#m_train = pd.concat([m_train, class_dummy_train], axis = 1)

#m_train = m_train.drop(["Pclass"], axis = 1)



#m_test = pd.concat([m_test, class_dummy_test], axis = 1)

#m_test = m_test.drop(["Pclass"], axis = 1)



#m_train.sample(5)
# Drop non-numeric categorical variables

train = train.drop(["Embarked","Sex", "Age", "SibSp", "Parch", "Fare"], axis = 1)

test = test.drop(["Embarked","Sex", "Age", "SibSp", "Parch", "Fare"], axis = 1)
train.sample(5)
test.sample(5)
# Check dataset status before modelling

train.info()

print('***************************************************')

test.info()
# As inspired by Nadin, we will use 80% of the data for training,

# and the rest 20% to test the accuracy of the model



predictors = train.drop(["Survived"], axis = 1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

# y_pred = gaussian.predict(x_val)

acc_gaussian = gaussian.score(x_val, y_val)

acc_gaussian
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

#y_pred = logreg.predict(x_val)

acc_logreg = logreg.score(x_val, y_val)

acc_logreg
# Support Vector Machine

svc = SVC()

svc.fit(x_train, y_train)

#y_pred = logreg.predict(x_val)

acc_svc = svc.score(x_val, y_val)

acc_svc
# Decision Tree Classifier

decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

#y_pred = decisiontree.predict(x_val)

acc_decisiontree = decisiontree.score(x_val, y_val)

acc_decisiontree
# Random Forest Classifier

randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

#y_pred = randomforest.predict(x_val)

acc_randomforest = randomforest.score(x_val, y_val)

acc_randomforest
# K-Nearest Neighbors

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

#y_pred = knn.predict(x_val)

acc_knn = knn.score(x_val, y_val)

acc_knn
# Generate Predictions

prediction = randomforest.predict(test)



submission_titanic = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': prediction })

#print(submission_titanic)

submission_titanic.to_csv("submission_titanic.csv", index = False)