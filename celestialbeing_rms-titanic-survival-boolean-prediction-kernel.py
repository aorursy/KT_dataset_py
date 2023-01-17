# Import needed tools

import os

import numpy as np

import pandas as pd

print(os.listdir("../input"))



# Import visualization libs

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Read the data sets Make a copy so that

# there will be no changes on the original data

read_train = pd.read_csv("../input/train.csv")

read_test  = pd.read_csv("../input/test.csv")

train = read_train.copy()

test  = read_test.copy()
# Explore each data set

print(train.columns)

train.shape
print(test.columns)

test.shape
train.describe()
test.describe()
train.sample(5)
test.sample(5)
# The idea here is I need to know first what are the

# data types of all the available features on the train data



# Survived: int

# Pclass: int

# Name: string

# Sex: string

# Age: float

# SibSp: int

# Parch: int

# Ticket: string

# Fare: float

# Cabin: string

# Embarked: string
# On the training data set, I want to know 

# how many and on what fields do nulls exist

print(pd.isnull(train).sum())
# Given that the total passengers on the training data set

# is 891, I think I'll just drop Cabin since it will be of no use

# I will retain the Age as I think it would be an important factor

# in terms of survivability

train = train.drop("Cabin", axis=1)

print(train.columns)

train.sample(3)
# Now I need to investiate what features are worth keeping or I could use

# to help me have a meaningful survivability prediction. I'll start with Age

# (just to prove if it is worth keeping)

sns.barplot(x="Age", y="Survived", data=train)
# Even if the graph will give some few insights, it is really hard 

# to understand due to the feature's high cardinality. Will try to 

# make age groups instead. Also I will process both train and test 

# here since I will impute missing values

train["Age"] = train["Age"].fillna(-0.5)

test["Age"]  = test["Age"].fillna(-0.5)

bins   = [-1, 0, 7, 13, 18, 25, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup']  = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)
# Hmm.. lucky babies :D, now 

# Checking survivability by sex

sns.barplot(x="Sex", y="Survived", data=train)
# Hmm.. more than 70% of female survived. To get a more accurate count

print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
# How about Pclass? based on the data dictionary of this data set, 

# this is more of socioeconomic class.. 1 being the highest and 3 being 

# the lowest. Checking survivability in terms of Pclass

sns.barplot(x="Pclass", y="Survived", data=train)
# And here we go, looks like the higher you socioeconomic status is, 

# the more likely you will survive from this disaster.. this is getting me sick

# I will now check the relationship between survivability and 

# the number of siblings/spouse aboard, as well as number of parent and children

sns.barplot(x="SibSp", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)
# I do not know with this data but for some reason, 

# only few people who does not have relatives aboard actually survived

# compared to this people with one or two relatives.. okay time to 

# bring some relatives now :D
# Revisiting our train and test data set, since we are now done with 

# our exploratory data analysis, it is time to clean the data and prepare it

# for consumption

print(train.shape)

print(test.shape)

print(train.columns)

print(test.columns)
# I have already dropped the cabin feature earlier for the training data set, 

# I will do the same on the test data set. Also I will remove ticket, fare and name

# features as I don't think that it will somehow help you survive anyway. It just 

# doesn't make any sense. I will also remove AgeGroup since we just used it actually 

# prove that Age feature is worth keeping

test  = test.drop(["Cabin","Fare","Ticket", "Name", "AgeGroup"], axis=1)

train = train.drop(["Fare","Ticket", "Name", "AgeGroup"], axis=1)

print(train.columns)

print(test.columns)
# Let's find all missing values on this data sets

print(pd.isnull(train).sum())

print(pd.isnull(test).sum())
# My train data does have 2 null Embarked features

print(train["Embarked"].unique())
# Since these feature has only 3 cardinality, I will just find the 

# Embarked value with the most frequency

print(train["Embarked"].count())

print(train[train["Embarked"] == "S"].count())

print(train[train["Embarked"] == "C"].count())

print(train[train["Embarked"] == "Q"].count())
# It pretty shows that S is dominant, so I will just fill all nulls (2) with S

train = train.fillna({"Embarked": "S"})
print(pd.isnull(train).sum())

print(pd.isnull(test).sum())
# All null values are now gone, splitting the train data set 

# between targets and features.

# I will use 80% of traning data set for training then 20% for validation

from sklearn.model_selection import train_test_split

target   = train["Survived"]

features = train.drop(['Survived', 'PassengerId'], axis=1)

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size = 0.20, random_state = 0)
# Cheking our training and validation data sets

print(x_train.sample(5))

print(x_val.sample(5))

print(y_train.sample(5))

print(y_val.sample(5))
# Checking all data types on the data set

for x in x_train.dtypes:

    print(x)

print("---")

for x in x_train.columns:

    print(x)
# The plan is to use different (with sense) algorithms/models to find the best 

# solution to predict survivability. Also this time, instead of mean absolute error

# or mean square error, I will just use a simple accuray scorer per algorithm.

# It would be better to just create a pipe and a reusable function to achieve this



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score



def model_scorer(x_train, x_val, y_train, y_val, model):

    numerical_transformer = SimpleImputer(strategy='constant')

    object_transformer    = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    

    # Define numerical features and categorical features from the dataframes

    object_cols    = x_train.select_dtypes(include=['object']).columns

    numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

    

    # Complete all missing values from the data set by means of imputation

    # bundle all transformers using ColumnTransformer

    data_cleanser = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('obj', object_transformer, object_cols)

        ]

    )

    

    # Create a pipeline

    pipe = Pipeline(steps=[

        ('Cleanser', data_cleanser),

        ('Model', model)

    ])

    

    # Start model fitting and scoring

    pipe.fit(x_train, y_train)

    preds = pipe.predict(x_val)

    score = round(accuracy_score(preds, y_val) * 100, 2)

    return score
# Importing models which will be used for score comparison

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier



models = [GaussianNB(), LogisticRegression(),

         SVC(), LinearSVC(), Perceptron(), DecisionTreeClassifier(),

         RandomForestClassifier(), KNeighborsClassifier(),

         SGDClassifier(), GradientBoostingClassifier()]

model_labels = ["GaussianNB", "LogisticRegression",

         "SVC", "LinearSVC", "Perceptron", "DecisionTreeClassifier",

         "RandomForestClassifier", "KNeighborsClassifier",

         "SGDClassifier", "GradientBoostingClassifier"]
print(model_labels[0])
import warnings

warnings.filterwarnings('ignore')

i = 0

while i < len(model_labels):

    score = model_scorer(x_train, x_val, y_train, y_val, models[i])

    print("Score for " + model_labels[i] + " = " + str(score))

    i += 1
# Looks like gradient boosting classifier is the best model

# to use for our survivability prediction. Time to create the 

# predictions using this model and test.csv

# Checking the test data set 

test.sample(3)
# Making predictions using GradientBoostingClassifier

# Save predictions as csv

numerical_transformer = SimpleImputer(strategy='constant')

object_transformer    = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])

    

object_cols    = x_train.select_dtypes(include=['object']).columns

numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

    

data_cleanser = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('obj', object_transformer, object_cols)

    ]

)



pipe = Pipeline(steps=[

    ('Cleanser', data_cleanser),

    ('Model', GradientBoostingClassifier())

])



ids = test['PassengerId']

pipe.fit(x_train, y_train)

preds  = pipe.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': preds })

#output.to_csv('prediction_results.csv', index=False)