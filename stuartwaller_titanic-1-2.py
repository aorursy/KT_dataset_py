# linear algebra 

import numpy as np



# data processing 

import pandas as pd



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style 



# machine learning algorithms 

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.info()
train_df.describe()
train_df.columns
test_df.columns # same columns, but does not include "Survived"
train_df.head(8)
# shows us which columns are missing data and how much

missing = train_df.isnull().sum().sort_values(ascending=False) # ascending=False (Top > Bottom)

missing
women = train_df[train_df["Sex"] == "female"]

men = train_df[train_df["Sex"] == "male"]



a = len(women[women["Survived"] == 1])

b = len(men[men["Survived"] == 1])

a, b # twice as many women survived as men - "Sex" is clearly important
women.describe() # 74% of women survived
men.describe() # only 18% of men survived
g = sns.relplot(x="Age", y="Survived", kind="line", data=men)

g.fig.autofmt_xdate()

"""

- :func:`scatterplot` (with ``kind="scatter"``; the default)

- :func:`lineplot` (with ``kind="line"``)

"""
h = sns.relplot(x="Age", y="Survived", kind="line", data=women)

h.fig.autofmt_xdate()
sns.barplot(x="Sex", y="Survived", data=train_df)
train_df.columns
train_df.Embarked.unique() # Oh right, the 2 NaN values
S_total = train_df[train_df["Embarked"] == "S"]

C_total = train_df[train_df["Embarked"] == "C"]

Q_total = train_df[train_df["Embarked"] == "Q"]

len(S_total), len(C_total), len(Q_total) # most people embarked from S, so most will die from S. Let's check the ratios, tho
S_live = len(S_total[S_total["Survived"] == 1])

C_live = len(C_total[C_total["Survived"] == 1])

Q_live = len(Q_total[Q_total["Survived"] == 1])



S_live, C_live, Q_live
round((S_live / len(S_total)), (3)), round((C_live / len(C_total)), (3)), round((Q_live / len(Q_total)), (3))
sns.barplot(x="Embarked", y="Survived", data=train_df)
train_df.columns
sns.barplot(x="Pclass", y="Survived", data=train_df)
train_df.Pclass.unique()
w_survived = women[women["Survived"] == 1]

m_survived = men[men["Survived"] == 1]
visualito =  sns.factorplot(x="Pclass", y="Survived", data=train_df, aspect=2)
women_Pclass_3 = len(women[women["Pclass"] == 3])

men_Pclass_3 = len(men[men["Pclass"] == 3])



ws_Pclass_3 = len(w_survived[w_survived["Pclass"] == 3])

ms_Pclass_3 = len(m_survived[m_survived["Pclass"] == 3])



print("CLASS 3 TOTAL:", women_Pclass_3, men_Pclass_3, "women:men")

print("SURVIVED:", ws_Pclass_3, ms_Pclass_3, "women:men")
women_Pclass_2 = len(women[women["Pclass"] == 2])

men_Pclass_2 = len(men[men["Pclass"] == 2])



ws_Pclass_2 = len(w_survived[w_survived["Pclass"] == 2])

ms_Pclass_2 = len(m_survived[m_survived["Pclass"] == 2])



print("CLASS 2 TOTAL:", women_Pclass_2, men_Pclass_2, "women:men")

print("SURVIVED:", ws_Pclass_2, ms_Pclass_2, "women:men")
women_Pclass_1 = len(women[women["Pclass"] == 1])

men_Pclass_1 = len(men[men["Pclass"] == 1])



ws_Pclass_1 = len(w_survived[w_survived["Pclass"] == 1])

ms_Pclass_1 = len(m_survived[m_survived["Pclass"] == 1])



print("CLASS 1 TOTAL:", women_Pclass_1, men_Pclass_1, "women:men")

print("SURVIVED:", ws_Pclass_1, ms_Pclass_1, "women:men")
train_df.columns
train_df.SibSp.unique() # of siblings / spouses aboard
train_df.Parch.unique() # of parents / children aboard
# creating "Relatives" column

train_df["Relatives"] = train_df["SibSp"] + train_df["Parch"]

train_df.columns
train_df = train_df.drop(["SibSp", "Parch"], axis = 1)
test_df["Relatives"] = test_df["SibSp"] + test_df["Parch"]

test_df.columns
test_df = test_df.drop(["SibSp", "Parch"], axis = 1)
train_df.info() # perfect, 891 values for "Relatives"
visual =  sns.factorplot(x="Relatives",y="Survived", data=train_df)
visual =  sns.factorplot(x="Relatives",y="Survived", data=train_df, aspect=2)
print(train_df.PassengerId.unique()[0:20], "etc...")
train_df.columns
train_df = train_df.drop(['PassengerId'], axis=1) # removing 'PassengerId' from dataset
train_df.columns
train_df.describe()
missing
train_df.Cabin
train_df = train_df.drop(['Cabin'], axis=1) # removing 'Cabin' from dataset
test_df = test_df.drop(['Cabin'], axis=1) # removing 'Cabin' from dataset
train_df.columns
test_df.columns
missing_updated = train_df.isnull().sum().sort_values(ascending=False) # ascending=False (Top > Bottom)

missing_updated
train_df.Age.describe()
mean = train_df["Age"].mean()

std = train_df["Age"].std() # how spread out our data is (measure of variation)

nansum = train_df["Age"].isnull().sum()



mean, std, nansum
np_test = np.random.randint(90, 100, size=10) # computes an array of 10 random integers between 90 and 100

np_test # this was a test
# mean = train_df["Age"].mean()

# std = train_df["Age"].std() # how spread out our data is (measure of variation)

# nansum = train_df["Age"].isnull().sum()



rand_age = np.random.randint(mean - std, mean + std, size = nansum)

rand_age
# code used to substitute the rand_age values in for the NaN values

train_df.loc[train_df.Age.isnull(), "Age"] = rand_age 
train_df.Age.describe()
train_df.Age.isnull().sum() # should equal 0
train_df.describe()
test_df.describe()
mean = test_df["Age"].mean()

std = test_df["Age"].std() # how spread out our data is (measure of variation)

nansum = test_df["Age"].isnull().sum()



rand_age = np.random.randint(mean - std, mean + std, size = nansum)

rand_age
test_df.loc[test_df.Age.isnull(), "Age"] = rand_age 
test_df.Age.isnull().sum()
test_df.describe()
test_df.Fare.isnull().sum()
train_df["Age"] = train_df["Age"].astype(int) # converting "Age"column to integer values

test_df["Age"] = test_df["Age"].astype(int)
train_df["Embarked"].describe()
train_df.loc[train_df.Embarked.isnull(), "Embarked"] = "S"
train_df["Embarked"].isnull().sum()
test_df.Embarked.describe()
test_df.loc[test_df.Embarked.isnull(), "Embarked"] = "S"
test_df["Embarked"].isnull().sum()
test_df.isnull().sum()
test_df["Fare"].describe()
test_df.loc[test_df.Fare.isnull(), "Fare"] = 35.627188
test_df.isnull().sum()
train_df.info()
train_df["Fare"] = train_df["Fare"].astype(int)

test_df["Fare"] = test_df["Fare"].astype(int)

test_df.info()
train_df["Name"].head(10)
# extract titles

train_df["Title"] = train_df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

train_df["Title"].describe()
# TRAINING DATA #



titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



# extract titles

train_df['Title'] = train_df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)



# replace titles with a more common title or as Rare

train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',

                                              'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')

train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')

train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')



# convert titles into numbers

# .map() used for substituting each value in a Series with another value from a dict 

train_df['Title'] = train_df['Title'].map(titles)



# filling NaN with 0 (to be safe)

train_df['Title'] = train_df['Title'].fillna(0)

train_df.columns
train_df["Title"] = train_df["Title"].astype(int)

train_df["Title"]
# TESTING DATA #



titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



# extract titles

test_df['Title'] = test_df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)



# replace titles with a more common title or as Rare

test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',

                                              'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')

test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')



# convert titles into numbers

# .map() used for substituting each value in a Series with another value from a dict 

test_df['Title'] = test_df['Title'].map(titles)



# filling NaN with 0 (to be safe)

test_df['Title'] = test_df['Title'].fillna(0)

test_df.columns
test_df["Title"] = test_df["Title"].astype(int)

test_df["Title"]
genders = {"male": 0, "female": 1}

train_df["Sex"] = train_df["Sex"].map(genders)

test_df["Sex"] = test_df["Sex"].map(genders)
train_df.Ticket.describe()
train_df = train_df.drop(["Ticket"], axis=1)

test_df = test_df.drop(["Ticket"], axis=1)
train_df.columns
test_df.columns
num_ports = {"S": 0, "C": 1, "Q": 2}

train_df["Embarked"] = train_df["Embarked"].map(num_ports)

test_df["Embarked"] = test_df["Embarked"].map(num_ports)
train_df["Embarked"].unique()
test_df["Embarked"].unique()
train_df.info()
train_df = train_df.drop(["Name"], axis=1)

test_df = test_df.drop(["Name"], axis=1)
train_df.info()
test_df.info()
# train_df minus the "Survived" feature

X_train = train_df.drop("Survived", axis=1) # axis=1 tells us that we want to drop a column



# only "Survived" (the target variable) from train_df

Y_train = train_df["Survived"] 



# copy of test_df without "PassengerId -> BECAUSE we removed "PassengerId" from train_df

# because, as we said before, we don't need to be testing with "PassengerId"

X_test = test_df.drop("PassengerId", axis=1).copy()



# X_train without "Survived" and X_test without "PassengerId" are practically the same content
# max_iter = The maximum number of passes over the training data (aka epochs)'

# tol = the stopping criterion



sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_prediction = sgd.predict(X_test) # what we would submit... printing Y_pred gives us all the 0s and 1s



print(sgd.score(X_train, Y_train))



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print(acc_sgd)



print(Y_prediction[0:10])
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



print(random_forest.score(X_train, Y_train))



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)



print(Y_prediction[0:10])
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_prediction = logreg.predict(X_test)



print(logreg.score(X_train, Y_train))



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(acc_log)



print(Y_prediction[0:10]) # just like random forest model, little flexibility 
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)



Y_prediction = knn.predict(X_test)



print(knn.score(X_train, Y_train))



acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(acc_knn)



print(Y_prediction[0:10]) # consistently hits 84.4
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)



Y_prediction = gaussian.predict(X_test)



print(gaussian.score(X_train, Y_train))



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print(acc_gaussian)



print(Y_prediction[0:10]) # stable, not jumping around
perceptron = Perceptron(max_iter=4)

perceptron.fit(X_train, Y_train)



Y_prediction = perceptron.predict(X_test)



print(perceptron.score(X_train, Y_train))



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print(acc_perceptron)



print(Y_prediction[0:10]) # stable response
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_prediction = linear_svc.predict(X_test)



print(linear_svc.score(X_train, Y_train))



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(acc_linear_svc)



print(Y_prediction[0:10])
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



Y_prediction = decision_tree.predict(X_test)



print(decision_tree.score(X_train, Y_train))



acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(acc_decision_tree)



print(Y_prediction[0:10])
# let's make a dataframe and find out

results = pd.DataFrame({

    "Model": ["Stochastic Gradient Descent", "Random Forest", "Logistic Regression", 

              "K Nearest Neighbor", "Gaussian Naive Bayes", "Perceptron",

              "Linear Support Vector Machine", "Decision Tree"], 

    "Score": [acc_sgd, acc_random_forest, acc_log, acc_knn, acc_gaussian, 

              acc_perceptron, acc_linear_svc, acc_decision_tree]})

results
results_df = results.sort_values(by="Score", ascending=False)

results_df
results_df.plot.bar(x="Model", y="Score")
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction_rf = random_forest.predict(X_test)



print(random_forest.score(X_train, Y_train))



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)



print(Y_prediction_rf[0:10])
import os

os.remove("/kaggle/working/wowsubmission.csv") # removed incorrect submission
output = pd.DataFrame({"PassengerId": test_df.PassengerId, "Survived": Y_prediction_rf})

output.to_csv("goodsubmission.csv", index=False)

print("Your submission was successfully saved!")
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_prediction_rf})
submission
submission.to_csv("wowsubmission.csv", index=False)
predictions = random_forest.predict(X_test)

pred_list = [int(x) for x in predictions]



test2 = pd.read_csv("/kaggle/input/titanic/test.csv")

output = pd.DataFrame({'PassengerId': test2['PassengerId'], 'Survived': pred_list})

output.to_csv('Titanic_with_logistic.csv', index=False)