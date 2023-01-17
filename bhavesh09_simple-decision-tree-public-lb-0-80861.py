import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn import preprocessing

from sklearn.tree import export_graphviz

%matplotlib inline
def score_in_percent (a,b):

    return (sum(a==b)*100)/len(a)
# This creates a pandas dataframe and assigns it to the train and test variables

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# store target as Y

Y_train = train["Survived"]

train.drop(["Survived"], axis=1, inplace=True)
#concat both datasets for ease of operation

num_train = len(train)

all_data = pd.concat([train, test])
# Populating null fare value with median of train set

all_data["Fare"]=all_data["Fare"].fillna(train["Fare"].median())

# Populating null age value with median of train set

#all_data["Age"]=all_data["Age"].fillna(train["Age"].median())

# Populating missing embarked with most frequent value - S

all_data["Embarked"]=all_data["Embarked"].fillna("S")

# Creating new feature as Title

all_data['Title'] = all_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Converting sex into binary

sex_mapping = {"male": 0, "female": 1}

all_data['Sex'] = all_data['Sex'].map(sex_mapping)
guess_age=all_data.groupby(['Title','Pclass','Sex'])['Age'].agg(['mean','count']).reset_index()

guess_age.columns= ['Title','Pclass','Sex','ga_mean','ga_cnt'] 

guess_age["ga_mean"]=guess_age["ga_mean"].fillna(28)

guess_age["ga_mean"]=guess_age["ga_mean"].astype(int)

all_data=all_data.merge(guess_age, how='left')

all_data.loc[(all_data.Age.isnull()),"Age"]=all_data[(all_data.Age.isnull())].ga_mean
# Drop columns which may cause overfit, also residual columns from above dataset

all_data.drop(["Cabin","Name","Ticket","PassengerId","ga_mean","ga_cnt"], axis=1, inplace=True)
# get dummies for categorical variables

all_data = pd.get_dummies(all_data)
X_train = all_data[:num_train]

X_test = all_data[num_train:]
#generating train and test splits for cross-validation

X_train, X_cv, y_train, y_cv = train_test_split( X_train, Y_train, test_size = 0.3, random_state = 100)
# Decision tree tuning

for crtr in ['gini','entropy']:

    for md in [3,4,5]:

        for spltr in ['best','random']:

            for mss in [6,10,16,26,42]:

                for msl in [6,10,16,26,42]:

                    dts = DecisionTreeClassifier(class_weight=None, criterion=crtr, max_depth=md,

                                max_features=None, max_leaf_nodes=None, min_samples_leaf=msl,

                                min_samples_split=mss, min_weight_fraction_leaf=0.0,

                                presort=False, random_state=100, splitter=spltr)

                    dts.fit(X_train, y_train)

                    y_pred = dts.predict(X_cv)

                    sip=score_in_percent(y_pred,y_cv)

                    print("score for {} criterion, {} max_depth, {} splitter, {} min_samples_split, {} min_samples_leaf is {}".format(crtr,md,spltr,mss,msl,sip))
clf_tuned = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,

                                max_features=None, max_leaf_nodes=None, min_samples_leaf=10,

                                min_samples_split=10, min_weight_fraction_leaf=0.0,

                                presort=False, random_state=100, splitter='random')

clf_tuned.fit(X_train, y_train)

y_pred = clf_tuned.predict(X_cv)

y_test_pred = clf_tuned.predict(X_test)

score_in_percent(y_pred,y_cv)
# This statement builds a dot file.

cols = list(X_train.columns.values)

tree.export_graphviz(clf_tuned, out_file='tunedtreewithdummies.dot',feature_names  = cols) 
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test_pred

    })

submission.to_csv('tunedtreewithdummies.csv', index=False) # LB : 0.74163