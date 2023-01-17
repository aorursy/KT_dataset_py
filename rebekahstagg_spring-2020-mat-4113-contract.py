# default libraries

import numpy as np

import pandas as pd

#code from the Beginner's Guide

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)



# libraries from the Beginner's Guide

import matplotlib as mpl

import matplotlib.cm as cm 

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings('ignore')

import string

import math

import sys





# default code to import datasets

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
# code (with some modifications) from the Beginner's Guide

figure, myaxis = plt.subplots(figsize=(10, 7.5))



sns.countplot(x = "Sex", 

                   hue="Survived",

                   data = train, 

                   linewidth=2, 

                   palette = {1:"forestgreen", 0:"gray"}, ax = myaxis)





## Fixing title, xlabel and ylabel

myaxis.set_title("Passenger Survival vs Sex", fontsize = 20)

myaxis.set_xlabel("Sex", fontsize = 15);

myaxis.set_ylabel("Number of Passengers Survived", fontsize = 15)

myaxis.legend(["Did not survive", "Survived"], loc = 'upper right')
# code used (with some modifications) from the Beginner's Guide

figure, embarked_bar = plt.subplots(figsize=(7, 7))



sns.barplot(x= train["Embarked"].value_counts().index, 

            y = train["Embarked"].value_counts(),

            palette = {'S':"teal", 'C':"coral", 'Q': "lightblue"},

            ax = embarked_bar)



embarked_bar.set_xticklabels(['Southampton', 'Chernboug', 'Queenstown'])

embarked_bar.set_ylabel('Frequency Count')

embarked_bar.set_title('Frequency of Port of Embarkment', fontsize = 16)
# code used (with some modifications) from the Beginner's Guide

figure, embarked_bar = plt.subplots(figsize = (8,10))

sns.barplot(x = "Embarked", 

            y = "Survived", 

            estimator = np.mean,

            data=train,

            palette = {'S':"teal", 'C':"coral", 'Q': "lightblue"},

            ax = embarked_bar,

            linewidth=2)

embarked_bar.set_title("Passenger Port of Embarkment Distribution - Survived vs Did not Survive", fontsize = 15)

embarked_bar.set_xlabel("Port of Embarkment", fontsize = 15);

embarked_bar.set_ylabel("% of Passenger Survived", fontsize = 15);
# code used (with some modifications) from the Beginner's Guide



figure, pclass_bar = plt.subplots(figsize = (8,10))

sns.barplot(x = "Pclass", 

            y = "Survived", 

            estimator = np.mean,

            data=train,

            palette = {1:"teal", 2:"coral", 3: "lightblue"},

            ax = pclass_bar,

            linewidth=2)



pclass_bar.set_title("Passenger Class Distribution - Survived vs  Did not Survive", fontsize = 15)

pclass_bar.set_xlabel("Passenger class (Pclass)", fontsize = 15);

pclass_bar.set_ylabel("% of Passenger Survived", fontsize = 15);

labels = ['Upper (1)', 'Middle (2)', 'Lower (3)']



val = sorted(train.Pclass.unique())

# val = [0,1,2] ## this is just a temporary trick to get the label right. 

pclass_bar.set_xticklabels(labels);
# code used (with some modifications) from the Beginner's Guide



# Explore Fare distibution

figure, myaxis = plt.subplots(figsize=(15, 7))



preimputation=sns.kdeplot(data=train["Fare"][(train["Survived"] == 0) & (

    train["Fare"].notnull())], kernel='gau', ax=myaxis, color="gray", shade=True, legend=True)



preimputation=sns.kdeplot(data=train["Fare"][(train["Survived"] == 1) & (

    train["Fare"].notnull())], kernel='gau', ax=myaxis, color="forestgreen", shade=True, legend=True)



myaxis.set_xlabel("Fare")

myaxis.set_ylabel("Probability Density")

myaxis.legend(["Did not survive", "Survived"], loc='upper right')

myaxis.set_title("Superimposed KDE plot for Fare and Survival",

                 loc='center', fontdict={'fontsize': 15}, color='black')
# code used from the Beginner's Guide

print('Train columns with null values:',train.isnull().sum(), sep = '\n')

print("-"*42)





print('Test/Validation columns with null values:', test.isnull().sum(),sep = '\n')

print("-"*42)
# code used from the Beginner's Guide

test[test['Fare'].isnull()]

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
# code used (with some modifications) from the Beginner's Guide

print('Number of missing values in Fare column for test set:', test.Fare.isnull().sum())
# code used (with some modifications) from the Beginner's Guide

train[train['Embarked'].isnull()]

train['Embarked'] = train['Embarked'].fillna('S')
# code used (with some modifications) from the Beginner's Guide

print('Number of missing values in Embarked column for train set:', test.Embarked.isnull().sum())
# First, we import the appropriate library

from sklearn.ensemble import RandomForestClassifier



# Next, we set our dependent variable as survival in the training dataset

y = train["Survived"]



# We choose which features (or rather variables) we want to consider in our model based on our EDA

features = ["Sex", "Embarked", "Pclass", "Fare"]



# These lines ensure that categorical data in both datasets is one hot encoded so that the model can actually access it in a meaningful way

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



# This is the actual model. n_estimators is the number of trees, and max_depth is tree depth. Specifying a random state ensures that each

# tree begins in the same place.

model = RandomForestClassifier(n_estimators=125, max_depth=5, random_state=1)

# Here we fit the model using our training data and its survival values

model.fit(X, y)

# Here we ask the model to use the test dataset features to predict survival and store those predictions in a variable

predictions = model.predict(X_test)



# Now we put our predictions in a dataframe, with the passenger ID of the test passengers corresponding to their predicted survival value

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

# We make our output a CSV file and choose its name

output.to_csv('contract_submission.csv', index=False)

# And finally, we print a message so we know that the model ran all the way through

print("Your submission was successfully saved!")