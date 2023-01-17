import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#libraries 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix



#models 

from sklearn.ensemble import RandomForestClassifier



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")



# pclass:    Ticket class 

# sibsp:     siblings / spouses  in the ship

# parch:     parents / children in the ship

# cabin:     Cabin number

# embarked:  Port of Embarkation
titanic_train.describe()
# Extracting  sign of respect(Mr, Mrs, Miss, Master)

#Function to Extract status ( need to pass exact colomn to function in dataframe)

import re

def Ext(col):

    #col = titanic_train['Name']

    names = []

    for val in col:

        names.append(val)

    def splitter(name): 

        result = re.split('(?<!\d)[,.]|[,.](?!\d)', name)

        result = result[1]

        return result

    final_res = []

    for name in names:

        add = splitter(name)

        final_res.append(add)

    return final_res 

titanic_train['Status'] = Ext(titanic_train["Name"])

titanic_test['Status'] = Ext(titanic_test["Name"])

# titanic_train['status'] = final_res

# titanic_train['status'].value_counts()
titanic_train['Status'].value_counts()
titanic_test['Status'].value_counts()
titanic_train['Survived'].value_counts()
#checking for missing values 

missing_total = titanic_train.isnull().sum()

missing_total
#plotting 

#attributes = ['Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']

corr_matrix = titanic_train.corr()

corr_matrix['Survived'].sort_values(ascending=False)
sns.barplot(x='SibSp', y='Survived', data=titanic_train)
sns.barplot(x='Parch', y='Survived', data=titanic_train)
Grid = sns.FacetGrid(titanic_train, row='Embarked', size=4.5, aspect=1.6)

Grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

Grid.add_legend()
grid = sns.FacetGrid(titanic_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# Trying out attribute combinations 

titanic_train['SibSpParch'] = (titanic_train['SibSp'] + titanic_train['Parch']) % 2

#titanic_train.corr()['Survived']['SibSpParch']

corr_matrix = titanic_train.corr()

corr_matrix['Survived'].sort_values(ascending=False)
titanic_test['SibSpParch'] = (titanic_test['SibSp'] + titanic_test['Parch']) % 2
# Dropping out useless features 

titanic_train_labels = titanic_train['Survived'].copy()

titanic_train = titanic_train.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)

titanic_test = titanic_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#Too much missing values in Cabin but here a cabin number looks like ‘C123’ and the letter refers to the deck

#so you can try to find something which makes sense using Cabin 

titanic_train.head()
# fill missing values in Embarked using common value

titanic_train['Embarked'].describe()

# since common values is S 

common ='S'

datasets = [titanic_train, titanic_test]

for val in datasets:

    val['Embarked']=val['Embarked'].fillna(common)

# titanic_test.isnull().sum()

# titanic_train.isnull().sum()
# Fill Age with median 

median = titanic_train['Age'].median()

titanic_train = titanic_train.fillna(median)

titanic_test = titanic_test.fillna(median)
# 1. Sex

gender = {"male":1, "female":0}

datasets = [titanic_train, titanic_test]

for val in datasets:

    val['Sex'] = val['Sex'].map(gender)
# 2. Embarked and Status  

# Here you can use sklearn one hot encoder or pandas get dummies 

titanic_train = pd.get_dummies(titanic_train)

titanic_test = pd.get_dummies(titanic_test)
titanic_train.describe()
# Filling train and test set with unavlible features 

titanic_train['Status_ Dona'] = 0 

features = ['Status_Major','Status_Mlle','Status_Jonkheer','Status_Don','Status_Capt','Status_Mme','Status_Sir','Status_the Countess','Status_Lady']

for ft in features:

    titanic_test[ft] = 0 



titanic_test.describe()
# Put feature names into a order 

titanic_train = titanic_train.sort_index(axis=1)

titanic_test = titanic_test.sort_index(axis=1)
#Standardization 

from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import MinMaxScaler

my_pip = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler()),

    #('min_max', MinMaxScaler())

])

titanic_train[['Pclass','Age','Fare','SibSp','Parch']] = my_pip.fit_transform(titanic_train[['Pclass','Age','Fare','SibSp','Parch']])

titanic_test[['Pclass','Age','Fare','SibSp','Parch']] = my_pip.fit_transform(titanic_test[['Pclass','Age','Fare','SibSp','Parch']])

titanic_train
model = RandomForestClassifier(n_estimators=100)

#model = DecisionTreeClassifier()

#model = KNeighborsClassifier(n_neighbors = 3)

model.fit(titanic_train, titanic_train_labels)

score = round(model.score(titanic_train, titanic_train_labels)*100, 2)

score
pred = model.predict(titanic_test)
# final_dframe = pd.DataFrame()

# titanic_test = pd.read_csv("test.csv")

# final_dframe['PassengerId'] = titanic_test['PassengerId']

# final_dframe['Survived'] = pred

# print(final_dframe)

#final_dframe.to_csv('sub_9_24.csv',index=False)