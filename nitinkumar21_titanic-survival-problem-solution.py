# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas packages

import numpy as np
import pandas as pd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Import all the files
gender = pd.read_csv("../input/gender_submission.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
train.tail()
# About the dataset
print(train.shape)
print(train.info())
# code for column-wise null count here

train.isnull().sum()
# code for row-wise null count here
train.isnull().sum(axis=1).head()
# code for column-wise null percentages here
round(100*(train.isnull().sum()/len(train.index)), 2)
#Handle Missing Values
age_by_sex=train.groupby(['Sex'])
age_by_sex
age_mean = age_by_sex['Age'].mean()
age_mean
#Handle the NAN value in age
female=train.loc[train['Sex']=='female']
female['Age'].replace(np.nan,27.9,inplace= True)
female.head()

male=train.loc[train['Sex']=='male']
male['Age'].replace(np.nan,30.7,inplace= True)
male.head()

train=pd.concat([male,female])
train.head()
# code for column-wise null percentages here
round(100*(train.isnull().sum()/len(train.index)), 2)
#Handle the NAN value in Embarked
seg_by_embarked=train.groupby(['Embarked'])
seg_by_embarked
embarked = seg_by_embarked['Embarked'].count()
embarked
# Replace the missing value with 'S'
train['Embarked'].replace(np.nan,'S',inplace= True)
round(100*(train.isnull().sum()/len(train.index)), 2)

# Describe The Train Dataset...
train.describe()
len(train.loc[train.Ticket.duplicated()].sort_values(by= 'Ticket'))
##Drop unnecessary coloumns which will not correlate to survival
## Cabin and Tickets are dropped
train=train.drop(columns=['Cabin','Ticket'])
train
# Correlation between Pclass and Survived
#Observation-PClass 1 has a very high survival rate of 62%

grp_by_pclass=train.groupby('Pclass')
grp_by_pclass['Survived'].mean()
# Correlation between Gender and Survived
#Observation-Female has surviving rate of 74%
grp_by_sex=train.groupby('Sex')
grp_by_sex['Survived'].mean()
# Correlation between sibling and Survived
#Observation-53% for one sibling
grp_by_sibling=train.groupby('SibSp')
grp_by_sibling['Survived'].mean()
# Correlation between Parents and Survived
#Observation-60% for one parents 3

grp_by_parch=train.groupby('Parch')
grp_by_parch['Survived'].mean()
# Analyse Data by visualizing
#Observations: 
# infants till age 4 has high survival rate
# Most People are from age 15 to 35
# Old people have survived
# Between 15 to 25 did not survive

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#Observations
#Age vs Pclass and Survived
#Most passengers survived in Pclass-1
#Infants survived in both Pclass-2 and Pclass-3
#Less people survived in Pclass-3

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
#Observations
# In all thee embarked female has higher survival rate
# PClass 1 has highest survival rate for both the gender except in Embarked=Q where PClass=3 Male has high survival
# rate

grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')
grid.add_legend()
#Observations
# Passengers with higher fare has higher survival rate

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Fare', alpha=.5, ci=None)
grid.add_legend()
#Drop Ticket and Caben in test data as well
test=test.drop(columns=['Cabin','Ticket'])
test.head()
# code for column-wise null percentages here
round(100*(test.isnull().sum()/len(test.index)), 2)
# Handle the missing age value
age_by_sex1=test.groupby(['Sex'])
age_by_sex1
age_mean1 = age_by_sex1['Age'].mean()
age_mean1


# Replace NaN Values
#Handle the NAN value in age

test['Age'].replace(np.nan,30.27,inplace= True)
test.head()
#Handle the NAN value in Fare
seg_by_fare=test.groupby(['Pclass'])
seg_by_fare
Fare = seg_by_fare['Fare'].mean()
Fare

fare_miss=test.loc[test['Fare'] == 0]
fare_miss
#Replace with PClass average Fare

test['Fare'].replace(0.0,94.28,inplace= True)



test.loc[test['Fare'] ==0]

round(100*(test.isnull().sum()/len(test.index)), 2)
train.head()
#Create title in both the datasets

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

train['Title']=train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir','Jonkheer','Dona'], 'Rare')

train['Title']=train['Title'].replace('Mlle', 'Miss')
train['Title']=train['Title'].replace('Ms', 'Miss')
train['Title']=train['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(test['Title'], test['Sex'])

test.head()
test['Title']=test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir','Jonkheer','Dona'], 'Rare')

test['Title']=test['Title'].replace('Mlle', 'Miss')
test['Title']=test['Title'].replace('Ms', 'Miss')
test['Title']=test['Title'].replace('Mme', 'Mrs')
    
pd.crosstab(test['Title'], test['Sex'])
#Title Mapping
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in (train,test):
    dataset['Title'] = dataset['Title'].map(title_mapping)
    #dataset['Title'] = dataset['Title'].fillna(0)

train.head()
test.head()
train.head()

#drop Name from train and test
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
test.head()
train.head()
# Convert Sex from categorical to numeric features
for dataset in (train,test):
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()
test.head()
test.head()
#Create Age band
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in (train,test):    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']=4
train.head()
test.head()




test.head()
train.head()
train.drop(columns=['AgeBand'])
#Create New feature Family Size and IsAlone
for dataset in (train,test):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in (train,test):
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#Drop other columns
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
train.head()
train.head()
train = train.drop(['AgeBand'], axis=1)
train.head()

for dataset in (train,test):
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
test.head()
train.head()
#Converting Embarked to Numeric Var
for dataset in (test,train):
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test.head()
train.head()
#Create Fare Band
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in (test,train):
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
 



train = train.drop(['FareBand'], axis=1)
train.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in (test,train):
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
round(100*(test.isnull().sum()/len(test.index)), 2)
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
Y_train.head()
X_train.head()
X_test.head()
#Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    
    })
submission
len(submission.loc[submission['Survived']==1])
submission.to_csv('/Users/nitinreshu/Documents/PGDDS/Kaggle/Titanic/submission.csv', index=False)
