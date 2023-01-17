# Data Analysis Libraries;

import numpy as np

import pandas as pd



# Data Visualization Libraries;

import matplotlib.pyplot as plt

import seaborn as sns



# To Ignore Warnings;

import warnings

warnings.filterwarnings('ignore')



# To Display All Columns:

pd.set_option('display.max_columns', None)



from sklearn.model_selection import train_test_split, GridSearchCV
# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



# Model Selection

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import metrics

from sklearn.preprocessing import StandardScaler,minmax_scale

# Read train and test data with pd.read_csv():

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# Copy data in order to avoid any change in the original:

train = train_data.copy()

test = test_data.copy()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
import missingno as msno
msno.bar(train);
train.head()
train.tail()
train.info()
test.head()
test.tail()
test.info()
train.describe().T
test.describe().T
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Ticket'].value_counts()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
sns.barplot(x='Pclass', y ='Survived',data = train);
sns.barplot(x = 'SibSp', y = 'Survived', data = train);
sns.barplot(x = 'Parch', y = 'Survived', data = train);
sns.barplot(x = 'Sex', y = 'Survived', data = train);
train.head()
test.head()
# We can drop the Ticket feature since it is unlikely to have useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)



train.head()
train.describe([0.10,0.25,0.50,0.75,0.90,0.99]).T
# It looks like there is a problem in Fare max data.Visualize with boxplot.

sns.boxplot(x = train['Fare']);
Q1 = train['Fare'].quantile(0.25)

Q3 = train['Fare'].quantile(0.75)

IQR = Q3 - Q1



lower_limit = Q1-1.5*IQR

lower_limit



upper_limit = Q3 + 1.5*IQR

upper_limit
# Observations with Fare data higher than the upper limit.



train['Fare'] > (upper_limit)
train.sort_values("Fare",ascending = False).head()
train.sort_values("Fare",ascending = False).tail()
# In boxplot, there are too many data higher than upper limit; we can not change all. Just repress the highest value -512-

train['Fare'] = train['Fare'].replace(512.3292, 312)
train.sort_values("Fare", ascending = False).head()
train.sort_values("Fare", ascending = False).tail()
test.sort_values("Fare", ascending = False)
test['Fare'] = test['Fare'].replace(512.3292, 312)
test.sort_values("Fare", ascending = False)
train.isnull().values.any()
train.isnull().sum()
train["Age"].fillna(0, inplace = True)
train.isnull().sum()
train["Cabin"].fillna(0, inplace = True)
train.isnull().sum()
100 * train.isnull().sum() / len(train)
train["Age"] = train["Age"].fillna(train["Age"].mean())
test["Age"] = test["Age"].fillna(test["Age"].mean())
train.isnull().values.any()
test.isnull().values.any()
train.isnull().sum()
test.isnull().sum()
train.isnull().sum()
test.isnull().sum()
train["Embarked"].value_counts()
# Fill NA with the most frequent value:

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
test[test["Fare"].isnull()]
test[["Pclass","Fare"]].groupby("Pclass").mean()
test["Fare"] = test["Fare"].fillna(12)
test["Fare"].isnull().sum()
# Create CabinBool variable which states if someone has a Cabin data or not:



train["CabinBool"] = train["Cabin"].isnull().astype('int')

test["CabinBool"] = test["Cabin"].isnull().astype('int')



train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)



train.head()
train.isnull().sum()
test.isnull().sum()
# Map each Embarked value to a numerical value:



embarked_mapping = {"S": 1, "C":2, "Q":3}



train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)
train.head()
# Convert Sex values into 1-0:



from sklearn import preprocessing



lbe = preprocessing.LabelEncoder()



train["Sex"] = lbe.fit_transform(train["Sex"])

test["Sex"] = lbe.fit_transform(test["Sex"])
train.head()
train["Title"] = train["Name"].str.extract('([A-Za-z]+)\.', expand = False)



test["Title"] = test["Name"].str.extract('([A-Za-z]+)\.', expand = False)
train.head()
train['Title'].value_counts()
train['Title'] = train['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')



train['Title'] = train['Title'].replace(['Countess','Lady','Sir'], 'Royal')



train['Title'] = train['Title'].replace('Mlle','Miss')



train['Title'] = train['Title'].replace('Ms','Miss')



train['Title'] = train['Title'].replace('Mme','Mrs')
test['Title'] = test['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')



test['Title'] = test['Title'].replace(['Countess','Lady','Sir'], 'Royal')



test['Title'] = test['Title'].replace('Mlle','Miss')



test['Title'] = test['Title'].replace('Ms','Miss')



test['Title'] = test['Title'].replace('Mme','Mrs')
train.head()
test.head()
train[["Title","PassengerId"]].groupby("Title").count()
train[['Title','Survived']].groupby(['Title'], as_index = False).agg({"count","mean"})
# Map each of the title groups to a numerical value



title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Royal":5, "Rare":5}



train['Title'] = train['Title'].map(title_mapping)
train.isnull().sum()
test['Title'] = test['Title'].map(title_mapping)
test.head()
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.head()
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

test['AgeGroup'] = pd.cut(test["Age"], bins,labels = mylabels)
# Map each Age value to a numerical value:

age_mapping = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test ['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
# Dropping the Age feature for now, might change:

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
train.head()
# Map Fare values into groups of numerical values:

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1,2,3,4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1,2,3,4])
# Drop Fare values:

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.head()
train.head()
train["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
# Create new feature of family size:



train['Single'] = train['FamilySize'].map(lambda s:1 if s == 1 else 0)

train['SmallFam'] = train['FamilySize'].map(lambda s:1 if s == 2 else 0)

train['MedFam'] = train['FamilySize'].map(lambda s:1 if 3 <= s <= 4 else 0)

train['LargeFam'] = train['FamilySize'].map(lambda s:1 if s>=5 else 0)
train.head()
# Create new feature of family size:



test['Single'] = test['FamilySize'].map(lambda s:1 if s == 1 else 0)

test['SmallFam'] = test['FamilySize'].map(lambda s:1 if s == 2 else 0)

test['MedFam'] = test['FamilySize'].map(lambda s:1 if 3 <= s <= 4 else 0)

test['LargeFam'] = test['FamilySize'].map(lambda s:1 if s >= 5 else 0)
test.head()
# Convert Title and Embarked into dummy variables:



train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"], prefix = "Em")
train.head()
test = pd.get_dummies(test, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Embarked"], prefix = "Em")
test.head()
train.groupby("Pclass")["Survived"].mean()
# Creat categorical values for Pclass:

train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"], prefix = "Pc")
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"], prefix = "Pc")
train.head()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X = train.drop(['Survived', 'PassengerId'], axis = 1)

Y = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.20, random_state = 17)
x_train.shape
x_train.head()
x_val.shape
x_val.head()
y_train.shape
y_train.head()
y_val.shape
y_val.head()