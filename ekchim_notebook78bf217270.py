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



import scipy.optimize as op



import matplotlib.pyplot as plt

import seaborn as sns

import re



from pandas import Series, DataFrame



from sklearn.feature_selection import SelectKBest, f_classif

from sklearn import ensemble

from sklearn.metrics import mean_squared_error



from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace

from pylab import scatter, show, legend, xlabel, ylabel, contour, title

from scipy.optimize import fmin_bfgs



# Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# Print to standard output, and see the results in the "log" section below after running your script

print("\n\nTop of the training data:")

print(train.head())



# print("\n\nSummary statistics of training data")

print(train.describe())



print("----------------------------")

# print data information

train.info()



sns.set_style('whitegrid')



# ----------------------------------------------------------------



# Before plotting the embarked data, we find that there are two missing values in the data.

# We can fill them with 'S' since it's the most occurring value.



train["Embarked"] = train["Embarked"].fillna("S")



# From plotting the Embarked data, we observe that most of the passengers embarked from Southampton.



fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(x='Embarked', data=train, palette="Set3", ax=axis1)



# Group and then plot the mean of survived passengers in Embarked. On average, least number of

# passengers from Southampton survived compared to the total number of embarked passengers.



embark_mean = train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()



# Assign integer values to each char value in Embarked for train data.



train.loc[train["Embarked"] == "S", "Embarked"] = 0

train.loc[train["Embarked"] == "C", "Embarked"] = 1

train.loc[train["Embarked"] == "Q", "Embarked"] = 2



# Assign integer values to each char value in Embarked for test data.



test.loc[test["Embarked"] == "S", "Embarked"] = 0

test.loc[test["Embarked"] == "C", "Embarked"] = 1

test.loc[test["Embarked"] == "Q", "Embarked"] = 2



# ----------------------------------------------------------------



fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))



# The Pclass plot shows that most passengers were from class 3.





# Group and plot the mean of survived passengers in Pclass. On average, first and second class

# passengers had higher survival rate than the passengers from the third class.



Pclass_mean = train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()



# ---------------------------------------------------------------

# Replace male and female occurrences with 0 and 1

train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"] == "female", "Sex"] = 1



test.loc[test["Sex"] == "male", "Sex"] = 0

test.loc[test["Sex"] == "female", "Sex"] = 1



# ---------------------------------------------------------------



fig_dims = (1, 1)



# Convert fare values from float to integer for train data.

train['Fare'] = train['Fare'].astype(int)



# The fare histogram shows that the fare price around 10 was the common price

# for most passengers.



# Fill missing fare values for test data with the corresponding median value, and then

# convert values from float to integer.

test['Fare'] = test['Fare'].fillna(test["Fare"].median())

test['Fare'] = test['Fare'].astype(int)



# -----------------------------------------------------------------



# Fill missing age values for the train and test data with corresponding median value,

# and convert values from float to integer.

train["Age"] = train["Age"].fillna(train["Age"].median())

train['Age'] = train['Age'].astype(int)



test["Age"] = test["Age"].fillna(test["Age"].median())

test['Age'] = test['Age'].astype(int)



# From the Age plot, we observe that passengers with Age < 20 had higher survival rate

# compared to passengers with Age > 20. That is child had better survival rate.



# -----------------------------------------------------------------



# We can generate a new feature "Family", which will depend on the passenger's siblings/spouse

# (SibSp) values and parent/child (Parch) values. For convinience, we will set 1

# for a passenger traveling with family member and 0 otherwise.

train["Family"] = train["SibSp"] + train["Parch"]

train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] = test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0



fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))



# Plot shows that more than a half of passengers were traveling alone.



# Average of survived shows that passengers traveling with family members

# had higer chanse for survival.

family_mean = train[["Family", "Survived"]].groupby(['Family'], as_index=False).mean()



# ----------------------------------------------------------------



# Generate new feature Namelength.

train["NameLength"] = train["Name"].apply(lambda x: len(x))

test["NameLength"] = test["Name"].apply(lambda x: len(x))



fig_dims = (1, 1)





# ----------------------------------------------------------------



# This function returns the title from a name.

def title(name):

    # Search for a title using a regular expression. Titles are made of capital and lowercase letters ending with a period.

    find_title = re.search(' ([A-Za-z]+)\.', name)

    # Extract and return the title If it exists.

    if find_title:

        return find_title.group(1)

    return ""





# Get all titles.

titles = train["Name"].apply(title)

titles2 = test["Name"].apply(title)



# Mapping possible titles to integer values. Some titles are compressed and share the same title codes since they are rare.

map_title = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,

             "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for i, j in map_title.items():

    titles[titles == i] = j

    titles2[titles2 == i] = j



# Add title values to corresponding column.

train["Title"] = titles

test["Title"] = titles2



pid = test["PassengerId"]



# ----------------------------------------------------------------



# Drop unnecessary features

# train = train.drop(['PassengerId','Name','Ticket', 'Cabin', 'SibSp','Parch'], axis=1)

# test = test.drop(['PassengerId','Name','Ticket', 'Cabin', 'SibSp','Parch'], axis=1)



train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)





# ----------------------------------------------------------------



def person(passenger):

    # define person based on age

    age, sex = passenger

    if age < 16:

        return 'child'

    else:

        return sex





train.loc[train["Sex"] == 0, "Sex"] = "male"

train.loc[train["Sex"] == 1, "Sex"] = "female"



test.loc[test["Sex"] == 0, "Sex"] = "male"

test.loc[test["Sex"] == 1, "Sex"] = "female"



train['Person'] = train[['Age', 'Sex']].apply(person, axis=1)

test['Person'] = test[['Age', 'Sex']].apply(person, axis=1)



# create dummy variables for Person column

person_dummy_titanic = pd.get_dummies(train['Person'])

person_dummy_titanic.columns = ['Child', 'Female', 'Male']



person_dummy_test = pd.get_dummies(test['Person'])

person_dummy_test.columns = ['Child', 'Female', 'Male']



train = train.join(person_dummy_titanic)

test = test.join(person_dummy_test)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))



sns.countplot(x='Person', data=train, palette="Set3", ax=axis1)



# Plot average of survived (male, female, or child)

person_mean = train[["Person", "Survived"]].groupby(['Person'], as_index=False).mean()



# Drop dummy variable Person

train.drop(['Person'], axis=1, inplace=True)

test.drop(['Person'], axis=1, inplace=True)



# Drop Male as they had lowest survival rate

train.drop(['Sex'], axis=1, inplace=True)

train.drop(['Male'], axis=1, inplace=True)



test.drop(['Sex'], axis=1, inplace=True)

test.drop(['Male'], axis=1, inplace=True)



# ----------------------------------------------------------------



predictors = ["Age", "Fare", "Embarked", "Family", "NameLength", "Title", "Child", "Female", "Pclass", "SibSp", "Parch"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(train[predictors], train["Survived"])



# Get the raw p-values for each feature, and transform them into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores and find which parameters ('Pclass', 'Age', e.t.c) are the best

fig_dims = (1, 1)





y_t = train['Survived']

X_t = train.drop("Survived", axis=1)



y = y_t.values

X_temp = X_t.values



X_temp_test = test.values



m, n = X_temp.shape



m_test, n_test = X_temp_test.shape



X = np.ones((m, n + 1))



X_test = np.ones((m_test, n_test + 1))



X[:, -n:] = X_temp



X_test[:, -n:] = X_temp_test



y.shape = (m, 1)



y_test = np.ones((m_test, 1))



# Initialize theta parameters

initial_theta = zeros(shape=(n + 1, 1))



#--------------------------------------------------------------

print("Для GradientBoost:")

params = {'n_estimators': 10000, 'min_samples_split': 3, 'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X, y)



result = clf.predict(X_test)

for i in range(0, result.shape[0]):

    if result[i] > 0.5:

        result[i] = 1

    else:

        result[i] = 0

result.shape = (m_test, 1)

#mse = mean_squared_error(y, clf.predict(X))

#print("ошибка: %.4f" % mse)



#p = prediction(array(optimal_theta), X)



#t = (p == y)



print(np.mean(t) * 100)

submission = pd.DataFramesubmission = pd.DataFrame({

    "PassengerId": pid,

    "Survived": result.reshape(-1)

})

submission.to_csv('titanic.csv', index=False)



#--------------------------------------------------------------



#y_test = prediction(optimal_theta, X_test)



#y_test = y_test.astype(int)



#submission = pd.DataFrame({

#    "PassengerId": pid,

#    "Survived": y_test.reshape(-1)

#})



#submission.to_csv('titanic.csv', index=False)




