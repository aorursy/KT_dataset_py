import os

import pickle

import numpy as np

import pandas as pd

from sklearn import tree

from sklearn import ensemble
#######################################

# https://www.kaggle.com/c/titanic/data

#######################################



datapath = "../input"



# load training data

df = pd.read_csv(os.path.join(datapath, 'train.csv'))

df.head()
new_df = df.copy()

del new_df['PassengerId']

del new_df['Name']

del new_df['Ticket']

del new_df['Cabin']



new_df.head()
# check if data contains missing values



pd.isnull(new_df).any()
# fill missing values 

print(new_df['Age'].mean())

new_df['Age'] = new_df['Age'].fillna(new_df['Age'].mean())



most_occuring = new_df['Embarked'].value_counts().index[0]

print(most_occuring)

new_df['Embarked'] = new_df['Embarked'].fillna(most_occuring)
# create dummy variables

dummy_df = pd.get_dummies(new_df, prefix=['sex', 'embarked'], drop_first=True)

dummy_df.head()
# split out data set



Y = dummy_df['Survived'].values

del dummy_df['Survived']

X = dummy_df.values



split_size = int(len(Y)*0.2)



train_x = X[split_size:]

train_y = Y[split_size:]

test_x = X[:split_size]

test_y = Y[:split_size]
# train our classifier



# clf = tree.DecisionTreeClassifier().fit(train_x, train_y)

clf = ensemble.RandomForestClassifier().fit(train_x, train_y)



print(clf.score(test_x, test_y))
# predicting submission data



pred_df = pd.read_csv(os.path.join(datapath, 'test.csv'))

pred_df.head()
# perform same operations as on training data



# remove some features

del pred_df['Name']

del pred_df['Ticket']

del pred_df['Cabin']



# fill missing values

for column in pred_df.columns:

    print(column, pred_df[column].isnull().any())
print(pred_df['Age'].mean())

pred_df['Age'] = pred_df['Age'].fillna(pred_df['Age'].mean())
print(pred_df['Fare'].mean())

pred_df['Fare'] = pred_df['Fare'].fillna(pred_df['Fare'].mean())
#create dummies



pred_df.head()

dummy_pred = pd.get_dummies(pred_df, prefix=['sex', 'embarked'], drop_first=True)

dummy_pred.head()
passenger_id = dummy_pred['PassengerId']

del dummy_pred['PassengerId']

pred_X = dummy_pred.values



pred_Y = clf.predict(pred_X)



out_df = pd.DataFrame()

out_df['PassengerId'] = passenger_id

out_df['Survived'] = pred_Y



out_df.head()

#out_df.to_csv(os.path.join(datapath, 'sub1.csv'), index=False)