import numpy as np

import pandas as pd

from IPython.display import display

import matplotlib.pyplot as plt

%matplotlib inline



test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')

combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data])
outcomes = train_data['Survived']

train = train_data.drop('Survived', axis=1)

train.head()
def accuracy_score(truth, pred):

    if len(truth) == len(pred):

        return 'Predictions have an accuracy of {:.2f}%'.format((truth == pred).mean()*100)

    else:

        return 'Number of predictions does not match the number of outcomes'
def predictions_0(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        predictions.append(0)

    return pd.Series(predictions)



predictions = predictions_0(train)

accuracy_score(outcomes, predictions)
def predictions_1(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        

        predictions.append( 0 if passenger['Sex'] == 'male' else 1)

        

    return pd.Series(predictions)



predictions = predictions_1(train)

accuracy_score(outcomes, predictions)
def accuracy_score(truth, pred):

    if len(truth) == len(pred):

        return 'Predictions have an accuracy of {:.2f}%'.format((truth == pred).mean()*100)

    else:

        return 'Number of predictions does not match the number of outcomes'
def predictions_0(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        predictions.append(0)

    return pd.Series(predictions)



predictions = predictions_0(train)

accuracy_score(outcomes, predictions)
def predictions_1(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        

        predictions.append(0 if passenger['Sex'] == 'male' else 1)

        

    return pd.Series(predictions)



predictions = predictions_1(train)

accuracy_score(outcomes, predictions)
def predictions_2(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        

        predictions.append(1 if passenger['Age'] < 10 or passenger['Sex'] == 'female' else 0)

        

    return pd.Series(predictions)



predictions = predictions_2(train)

accuracy_score(outcomes, predictions)
def predictions_3(data):

    

    predictions = []

    

    for _, passenger in data.iterrows():

        

        if (passenger['Sex'] == 'male'):

            if (passenger['Pclass'] == 2):

                if (passenger['Age'] > 15):

                    predictions.append(0)

                else:

                    predictions.append(1)

            else:

                predictions.append(0)

        else:

            if passenger['SibSp'] > 2:

                predictions.append(0)

            else:

                predictions.append(1)

        

    return pd.Series(predictions)



predictions = predictions_3(train)

accuracy_score(outcomes, predictions)
predictions = predictions_3(test_data)

predictions.head()
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

submission.to_csv('submission.csv', index=False)