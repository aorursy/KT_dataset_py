# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import re



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
# Genenrate training set and testing set

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#train = pd.read_csv('/Users/machen/Downloads/titanic data/train.csv')

#test = pd.read_csv('/Users/machen/Downloads/titanic data/test.csv')

full = train.append( test , ignore_index = True )
train.head()
full.describe()
# Pivot pclass and survival rate

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Pivot sex and survival rate

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Pivot sibling number and survival rate

sibsp_sur = train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=False)

sibsp_sur.plot(x='SibSp', y='Survived')
# Pivot parch number and survival rate

parch_sur = train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False)

parch_sur.plot(x='Parch', y = 'Survived')
## Pivot embark port and survival rate

train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Embarked', ascending=False)
# slicing data by embark and sex, people boarding from port C have higher survival, regardless of the sex. 

train[["Embarked", "Survived",'Sex']].groupby(['Embarked','Sex'], as_index=False).mean().sort_values(by='Embarked', ascending=False)
# slicing data by embark and class. Higher percentage of passengers are at class 1.

train[["Embarked", "Survived",'Pclass']].groupby(['Embarked','Pclass'], as_index=False).count().sort_values(by='Embarked', ascending=False)
# slicing data by embark and sex, people boarding from port C have higher survival, regardless of the sex. 

train[["Embarked", "Survived",'Pclass']].groupby(['Embarked','Pclass'], as_index=False).mean().sort_values(by='Embarked', ascending=False)
## plot histgram of age for each segment of embark data

## passengers embarking at port C have higher percentage of infants and elderly people, resulting in higher survival rate.

fig, (axis1,axis2, axis3) = plt.subplots(1,3,figsize=(15,4))

axis1.set_title('Age hist - embark S')

axis2.set_title('Age hist - embark Q')

axis3.set_title('Age hist - embark C')



train[train['Embarked']=='S']['Age'].hist(bins=70, ax=axis1)

train[train['Embarked']=='Q']['Age'].hist(bins=70, ax=axis2)

train[train['Embarked']=='C']['Age'].hist(bins=70, ax=axis3)
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\. ', name)

    if title_search:

        return title_search.group(1)

    return ""
full['title'] = full['Name'].apply(get_title)

full['title'].value_counts()
full['title'] = full['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

full['title'] = full['title'].replace('Mlle','Miss')

full['title'] = full['title'].replace('Ms','Miss')

full['title'] = full['title'].replace('Mme','Mrs')

full['title'].value_counts()
# average survived passengers by age, sliced by Sex

fig, (axis1,axis2) = plt.subplots(2,1,figsize=(30,16))

average_age = train[['Sex',"Age", "Survived"]].groupby(['Sex','Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age[average_age['Sex']=='female'],ax = axis1)

sns.barplot(x='Age', y='Survived', data=average_age[average_age['Sex']=='male'],ax = axis2)
full['Age'].hist(bins = 60)
#full['Fare'].hist(bins = 60)

full[full['Fare']<100]['Fare'].hist(bins = 20)
full.isnull().sum()
# Fill missing values of Age with the average of Age (median)

full[ 'Age' ] = full.Age.fillna( full.Age.median() )



# Fill missing values of Fare with the average of Fare (median)

full[ 'Fare' ] = full.Fare.fillna( full.Fare.median() )
full.set_value((full['Fare']<40) , 'fare_category','40-')

full.set_value((full['Fare']>40) & (full['Fare']<=100), 'fare_category','40-100')

full.set_value((full['Fare']>100) & (full['Fare']<=200), 'fare_category','200-300')

full.set_value((full['Fare']>200) , 'fare_category','300+')

full.head()
full.set_value(full['Age']<=12, 'age_category','childrean')

full.set_value((full['Age']>12) & (full['Age']<60), 'age_category','adult')

full.set_value(full['Age']>=60, 'age_category','elderly')

full.head()
# Transform Sex into binary values 0 and 1

sex = pd.Series(np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

sex.head()
# Create a new variable for every unique value of Pclass

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
# Create a new variable to represent whether one passenger has siblings or parches

full['family']= full[ 'Parch' ] + full[ 'SibSp' ] +1 #including the passenger self

family = pd.get_dummies(full['family'], prefix='Family Size')

family.head()
# Createa a new variable for every age category

age = pd.get_dummies(full['age_category'], prefix = 'Age')

age.head()
# Create a new variable for every fare category

fare = pd.get_dummies(full['fare_category'],prefix = 'Fare')

fare.head()
# Select which features/variables to include in the dataset:

# pclass , sex , family , fare, Age, 



full_X = pd.concat( [ pclass , sex , family , age, fare ] , axis=1 )

full_X.head()
# Create all datasets that are necessary to train, validate and test models

X_train = full_X[ 0:891 ]

Y_train = train.Survived

X_test = full_X[ 891: ]

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
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

    'Model': ['Logistic Regression', 

              'Support Vector Machines',

              'Stochastic Gradient Decent',

              'Decision Tree',

              'Random Forest'],

    'Score': [acc_log, acc_svc, acc_sgd, acc_decision_tree, acc_random_forest

              ]})

models.sort_values(by='Score', ascending=False)
## feature importance in random forest model

colnames = X_train.columns

importance_dic = dict(zip(colnames,random_forest.feature_importances_))

importance_dic
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)