# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
# Loading data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

passenger_Id = test['PassengerId']
train.head(5)
# Feature engineering

full_data = [train, test]

for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Creating features
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in full_data:
    
    # Feature that describes the person
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)   
    # Feature that tells if the person had a cabin
    dataset['has_cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    # Feature that tells the FamilySize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Feature that tells if a person was alone
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
train.head()
# Eliminating columns that will not be used (Feature Selection)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'] # Categorical Features that will not be part of prediction
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

train.head()
# Check which columns have null values
null_columns_train=train.columns[train.isnull().any()]
null_columns_test=test.columns[test.isnull().any()]

print(train[null_columns_train].isnull().sum())
print(test[null_columns_test].isnull().sum())
# Getting rid of nulls without changing data distribution
full_data = [train, test]

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S') # Fill it with the Mode
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) # Fill it with the Median
        
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
# Check which columns have null values
null_columns_train=train.columns[train.isnull().any()]
null_columns_test=test.columns[test.isnull().any()]

print(train[null_columns_train].isnull().sum())
print(test[null_columns_test].isnull().sum())
full_data = [train, test]

# Without null values, apply a grouping technique to Age and Fare
for dataset in full_data:    
    dataset.loc[dataset['Fare'] <= dataset['Fare'].quantile(0.2), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > dataset['Fare'].quantile(0.2)) & (dataset['Fare'] <= dataset['Fare'].quantile(0.4)), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > dataset['Fare'].quantile(0.4)) & (dataset['Fare'] <= dataset['Fare'].quantile(0.6)), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > dataset['Fare'].quantile(0.6)) & (dataset['Fare'] <= dataset['Fare'].quantile(0.8)), 'Fare'] = 3
    dataset.loc[dataset['Fare'] > dataset['Fare'].quantile(0.8), 'Fare'] = 4
    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# Mapping categorical features to integer values
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head(5)


# Checking correlation between features

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
features = train.drop("Survived", axis=1)
labels = train["Survived"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( features, labels, test_size=0.2, random_state=42)
# SVM

from sklearn.metrics import fbeta_score, accuracy_score
clf = SVC(kernel='rbf', C=10, gamma=0.6)
clf.fit(x_train, y_train)
print("Accuracy for training data")
preds = clf.predict(x_train)
print(accuracy_score(y_train, preds))
print("Accuracy for testing data")
preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
print("F-score")
print(fbeta_score(y_test, preds, beta=0.5))
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print("Accuracy for training data " + str(acc_random_forest))
print("Accuracy for testing data " + str(round(accuracy_score(y_pred, y_test) * 100, 2)))
print('F score: ' + str(fbeta_score(y_test, preds, beta=0.5)))
forest_features = random_forest.feature_importances_

cols = train.columns.values

# Scatter plot 
trace = go.Scatter(
    y = forest_features,
    x = cols,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = forest_features,
        colorscale='Portland',
        showscale=True
    ),
    text = cols
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
# Submission area

model_pred = clf.predict(test) # Test data provided by the competition for submission

submission = pd.DataFrame({
        "PassengerId": passenger_Id,
        "Survived": model_pred
    })
submission.to_csv('submission.csv', index=False)