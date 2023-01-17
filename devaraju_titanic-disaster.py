# Step:1  Import libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# Step:2 Read and Explore the data

# Load the data 

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# peep into the data

train.describe()
# peep into all data

train.describe(include = "all")
# Step:3 Primary Data analysis



# What are the features present in this dataset

train.columns
# Lets look at some sample values to get rough idea about each features

train.sample(10)
# Now lets do detailed analysis of features

train.describe(include = "all")
# Total count is 891 (total number of passengers) and age, cabin has some data missing 

(891-714)*100/891
(891-204)*100/891


# 19.86 % of age data is missing

# 77.10 % of cabin data is missing
# Check for alien and missing values

pd.isnull(train).sum()
# Step:4 Data visualization



# Check which sex survived the most

sns.barplot(x = "Sex", y = "Survived", data = train)
# Around 18% males and 75% females were survived

# Females are more likely to survive than males. Sex is essential feature
# Check what class of passengers survived the most

sns.barplot(x = 'Pclass', y = 'Survived', data = train)
# We can easily say that passengers travelling in Upper class are more likely to survive

# Upper class survivers: around 63%

# Middle class survivers: around 48%

# Lower class survivers: around 25%
# Check the impact of presence of siblings/spouse on chances of survival

sns.barplot(x = 'SibSp', y = 'Survived', data = train)
# People accompanying 1 or 2 family member are more likey to survive than others
# Check the impact of presence of parents/children on chances of survival

sns.barplot(x = 'Parch', y = 'Survived', data = train )
# This also reveals that people with 1-2-3 family members are more likely to survive
# Check the age group of people who survived

sns.barplot(x = 'Age', y = 'Survived', data = train)
# Looks awful

# Divide age into different groups

# Lets group be, Baby(0-5), Child(5-12), Teenager(12-18)

# Student(18-25), Young_Adult(25-35), Adult(35-60), Senior(60-100) 

def group_age(data):

    values = (0, 5, 12, 18, 25, 35, 60, 100) # range of values for which age gr

    group_name = ['Baby', 'Child', 'Teenager', 'Student', 'Young_Adult', 'Adult', 'Senior' ] # labels for resulting category

    category = pd.cut(data.Age, values, labels = group_name)  

    data.Age = category

    return(data)



group_age(train)

group_age(test)



sns.barplot( x = 'Age', y = 'Survived', data = train)
# Babies are more likely to survive than any other group
# Upper class people are more likey to stay in cabin and chances of surbival might be more

# But Cabin feature has many non-integer values 

train['Cabin_'] = train['Cabin'].notnull().astype('int')

test['Cabin_'] = test['Cabin'].notnull().astype('int')



sns.barplot(x = 'Cabin_', y = 'Survived', data = train)
# Step:5 Cleaning the data

# We'll drop some un-useful features

# Take a copy of each dataset

train2 = train.copy()

test2 = test.copy() 

train2
# Drop follwing features as they are not important

train2 = train2.drop(['Cabin', 'Ticket'], axis = 1)

test2 = test2.drop(['Cabin', 'Ticket'], axis = 1)



# Next we'll try to fill the missing features in Age and Embarked  
# Fill embarked feature with S, because its the maximum value

train2 = train2.fillna({'Embarked' : 'S'})
# Fill missing age

# Combine both age

combine = [train2, test2]



# Extract title for each name

for data in combine:

    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand = False)



pd.crosstab(train2['Title'], train2['Sex'])
train2
# Replace various titles with common names

for data in combine:

    data['Title'] = data['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace(['Countless', 'Lady', 'Sir'], 'Royal')

    data['Title'] = data['Title'].replace(['Mlle', 'Ms'],'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

train2[['Title', 'Survived']].groupby(['Title'], as_index = False).mean

    
train2
# map each of title to numeric value



mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Royal":5, "Rare":6}

for data in combine:

    data['Title'] = data['Title'].map(mapping)

train2.head()

# Fill the missing values with mode (yoy can also use mean, meadian etc)

mr_age = train2[train2['Title'] == 1]['Age'].mode()

miss_age = train2[train2['Title'] == 2]['Age'].mode()

mrs_age = train2[train2['Title'] == 3]['Age'].mode()

master_age = train2[train2['Title'] == 4]['Age'].mode()

royal_age = train2[train2['Title'] == 5]['Age'].mode()

rare_age = train2[train2['Title'] == 6]['Age'].mode()



age_mapping = {1:'Young_Adult', 2:'Student', 3:'Adult', 4:'Baby', 5:'Adult', 6:'Adult'}



for x in range(len(train2['Age'])):

    if pd.isnull(train2['Age'][x]):

        train2['Age'][x] = age_mapping[train2['Title'][x]]



for x in range(len(test2['Age'])):

    if pd.isnull(test2['Age'][x]):

        test2['Age'][x] = age_mapping[test2['Title'][x]]



train2.dtypes
pd.isnull(test2).sum()
pd.isnull(train2).sum()
# map age value to number

age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young_Adult':5, 'Adult':6, 'Senior':7}

train2['Age'] = train2['Age'].map(age_map)

test2['Age'] = test2['Age'].map(age_map)



train2.head()
# Drop the name feature

train2 = train2.drop(['Name'], axis = 1)

test2 = test2.drop(['Name'], axis = 1)
# map each Sex to number

sex_map = {'male':0, 'female':1}

train2['Sex'] = train2['Sex'].map(sex_map)

test2['Sex'] = test2['Sex'].map(sex_map)



train2.head()
# map each embarked value to categorical

embark_map = {'S':1, 'C':2, 'Q':3}

train2['Embarked'] = train2['Embarked'].map(embark_map)

test2['Embarked'] = test2['Embarked'].map(embark_map)

train2.head()
# Seperate fare into a group of values

for x in range(len(test2['Fare'])):

    if pd.isnull(test2['Fare'][x]):

        pclass = test['Pclass'][x]

        test2['Fare'][x] = round(train[train['Pclass'] == pclass]['Fare'].mean(),4)



train2['FareBand'] = pd.qcut(train2['Fare'], 4, labels = [1, 2, 3, 4])

test2['FareBand'] = pd.qcut(test2['Fare'], 4, labels = [1, 2, 3, 4])



# drop Fare feature

train2 = train2.drop(['Fare'], axis = 1)

test2 = test2.drop(['Fare'], axis = 1)
for x in range(len(train2['Title'])):

    if pd.isnull(train2['Title'][x]):

        train2['Title'][x] = 2

        print(train2['Title'][x])
# Its time for fenale

# Step6: Choose the best model



# split the train2 data into train, test split

from sklearn.model_selection import train_test_split



predictors = train2.drop(['Survived', 'PassengerId'], axis = 1)

target = train2['Survived']



x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20, random_state = 0)



# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

accuracy_nb = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_nb)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



lreg = LogisticRegression()

lreg.fit(x_train, y_train)

y_pred = lreg.predict(x_test)

accuracy_lr = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_lr)
# Support Vector Machines (SVM)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



svm = SVC()

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

accuracy_svm = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_svm)
# Linear SVC

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score



lsvc = LinearSVC()

lsvc.fit(x_train, y_train)

y_pred = lsvc.predict(x_test)

accuracy_lsvc = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_lsvc)
# Perceptron

from sklearn.linear_model import Perceptron

from sklearn.metrics import accuracy_score



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

accuracy_percep = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_percep)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



dtree = DecisionTreeClassifier()

dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)

accuracy_dt = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_dt)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rforest = RandomForestClassifier()

rforest.fit(x_train, y_train)

y_pred = rforest.predict(x_test)

accuracy_rf = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_rf)
# K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy_knn = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_knn)
# Stochastic Gradient Descent (SGD)

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

accuracy_sgd = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score



gboost = GradientBoostingClassifier()

gboost.fit(x_train, y_train)

y_pred = gboost.predict(x_test)

accuracy_gboost = round(accuracy_score(y_pred, y_test)*100)

print(accuracy_gboost)
# Lets look at accuracy of every model 

models = pd.DataFrame({'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Decsion Tree','Linear SVC', 'Stochastic Gradient Descent', 'Grdient Boosting Classifier'],

                      'Score':[accuracy_svm, accuracy_knn, accuracy_lr, accuracy_rf, accuracy_nb, accuracy_percep, accuracy_dt, accuracy_lsvc, accuracy_sgd, accuracy_gboost]})

models.sort_values(by = 'Score', ascending = False)

# Final Step: Create Submission File



ids = test2['PassengerId']

prediction = gboost.predict(test2.drop('PassengerId', axis = 1))



output = pd.DataFrame({'PassengerId':ids, 'Survived': prediction})

output.to_csv('submission.csv', index = False)