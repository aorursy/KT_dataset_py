import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import json

sns.set_style('darkgrid')
%matplotlib inline
# import original data

data = pd.read_csv('../input/games.csv')
data.head()
fig = plt.figure(figsize=(10,10))

sns.heatmap(data[['t1_towerKills','t1_inhibitorKills','t1_dragonKills','t1_baronKills',

                  't2_towerKills','t2_inhibitorKills','t2_dragonKills','t2_baronKills','winner']].corr(),annot=True,square=True)
from sklearn.model_selection import train_test_split
X = data[['t1_baronKills', 't1_dragonKills', 't2_baronKills', 't2_dragonKills']]

y = data['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
X = data[['t1_towerKills','t1_inhibitorKills','t2_towerKills','t2_inhibitorKills']]

y = data['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
def logRegModel(X, y):

    '''

    X: data used to predict outcome

    y: outcome data

    

    function used to split data into train and test sections,

    create a logistic regression model, and train the model

    with the train data. Finally, the function will print

    a classification report to results.

    '''



    # Split the data (X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)



    # Create and train model

    logmodel = LogisticRegression()

    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)



    # Evaluate the model

    print(classification_report(y_test, predictions))
X = data[['t1_baronKills', 't1_dragonKills', 't2_baronKills', 't2_dragonKills',

         't1_towerKills','t1_inhibitorKills','t2_towerKills','t2_inhibitorKills']]

y = data['winner']
logRegModel(X, y)