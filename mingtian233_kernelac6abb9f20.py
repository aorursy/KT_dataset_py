import pandas as pd
import numpy as np
import seaborn as sns
sns.set() #seaborn for plots
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('../input/train.csv');
realTest1 = pd.read_csv('../input/test.csv');
#test = pd.read_csv('input/test.csv');

#if households don't own tablets, the value was set to Nan
#this sets them to zero
train['v18q1'].fillna(0, inplace=True);
realTest1['v18q1'].fillna(0, inplace=True);

#if a household doesn't pay rent, the value is Nan
train['v2a1'].fillna(0, inplace=True);
realTest1['v2a1'].fillna(0, inplace=True);

#if the person is not behind in school years, the value is Nan
train['rez_esc'].fillna(0, inplace=True);
realTest1['rez_esc'].fillna(0, inplace=True);

dropColumns = [
    "Id",
    "idhogar",
    "edjefe",
    "edjefa",
    "dependency"
]

train = train.drop(dropColumns,axis=1);
train.fillna(0, inplace=True);

realTest2 = realTest1.drop(dropColumns,axis=1);
realTest2.fillna(0, inplace=True);

newTrain = train;
trainTarget = train;
trainTarget = trainTarget.loc[:,'Target':];
newTrain = newTrain.drop("Target",axis=1);

tree = DecisionTreeClassifier().fit(newTrain,trainTarget);

prediction = tree.predict(realTest2);

my_submission = pd.DataFrame({'Id': realTest1.Id, 'Target': prediction});
my_submission.to_csv('submission.csv', index=False);
