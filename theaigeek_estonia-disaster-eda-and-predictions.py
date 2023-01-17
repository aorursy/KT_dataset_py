import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

est_dis = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv") #est_dis indicates Estonia Disaster, however you can use **df** if it's confusing

est_dis
#Checking top 5 rows of our data

est_dis.head()
#Checking how many passenger survived(1) and non-survived(0)

est_dis.Survived.value_counts()
#Checking whether any missing data

est_dis.isnull().sum()
#let's find out survival percentage

surv_percnt = est_dis.Survived.value_counts()[1]/len(est_dis)*100

print('Percentage of survived passengers: ' "{:.2f}".format(surv_percnt)+'%')
est_dis.Category.value_counts()
#Now, let's check total number of male and females

est_dis.Sex.value_counts()
pd.crosstab(est_dis.Sex, est_dis.Survived)
survivedBySex = est_dis.groupby('Sex')['Survived'].mean()

survivedBySex
#plotting Survivability sex wise

%matplotlib inline

plt.style.use('seaborn-whitegrid')

fig , ax = plt.subplots(figsize=(10,6))

ax = survivedBySex.plot.bar()

ax.set(xlabel='Sex',

      ylabel='Survived',

      title='Survival rate by Sex');
survivedByAge = est_dis.groupby('Age')['Survived'].mean()

survivedByAge
#The above information wasn't so helpful but if we plot these data, it may make sense

fig, ax = plt.subplots(figsize=(10,6))

ax = survivedByAge.plot.bar()

ax.set(xlabel='Age',

      ylabel='Survived',

      title='Survival rate Age wise');
survivedByCategory = est_dis.groupby('Category')['Survived'].mean()

survivedByCategory
#Let's plot the above data

fig, ax = plt.subplots(figsize=(10,6))

ax = survivedByCategory.plot.bar()

ax.set(xlabel='Category',

      ylabel='Survived',

      title='Survival Category wise');
survivedByCountry = est_dis.groupby('Country')['Survived'].mean()

survivedByCountry
fig, ax = plt.subplots(figsize=(10,6))

ax = survivedByCountry.plot.bar()

ax.set(xlabel='Country',

      ylabel='Survived',

      title='Survival Country wise');
#let's drop Firstname, lastname, PassengerId columns

est_dis.drop(['PassengerId','Firstname','Lastname'], axis=1, inplace=True)
est_dis.head()
est_dis.info()
#using labelencoder to convert all strings into integers in the dataframe

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for item in list(est_dis.columns):

    if est_dis[item].dtype=='object':

        est_dis[item]= le.fit_transform(est_dis[item])
est_dis.info()
#splitting data into X and y

X = est_dis.drop('Survived', axis=1)

y = est_dis['Survived']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,

                                                test_size=0.2)
#importing all the models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
#putting all the models in a dictionary

models = {"LogisticRegression":LogisticRegression(),

         "KNeighboursClassifier": KNeighborsClassifier(),

         "RandomForestClassifier":RandomForestClassifier()}
#Creating a function to fit our data in models and evaluate score

def fit_score(models,X_train,X_test,y_train,y_test):

    np.random.seed(40) #so our results can be reproducable

    evaluate = {} #this empty list will contain our evaluated score

    for name, model in models.items():

        model.fit(X_train,y_train) #fitting trained data in a model

        evaluate[name]= model.score(X_test,y_test) #evaluate score on test data

    return evaluate
evaluate= fit_score(models=models,

                   X_train=X_train,

                   X_test=X_test,

                   y_train=y_train,

                   y_test=y_test)

evaluate
#Different logistic Regression parameters

param_grid = {"C": np.logspace(-4,4,20),

               "solver":["liblinear"]}

from sklearn.model_selection import GridSearchCV

np.random.seed(55)

grid_log_reg = GridSearchCV(LogisticRegression(),

                           param_grid=param_grid,

                           cv=5,

                           verbose=True)

grid_log_reg.fit(X_train,y_train)
grid_log_reg.best_params_
grid_log_reg.score(X_test,y_test)