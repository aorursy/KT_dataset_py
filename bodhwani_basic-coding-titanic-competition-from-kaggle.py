import numpy as np

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames



# Import supplementary visualizations code visuals.py

import visuals as vs

%matplotlib inline





# Load the dataset

train_file = 'train.csv'

test_file = 'test.csv'

train_full_data = pd.read_csv(train_file)

test_full_data = pd.read_csv(test_file)



#display(train_full_data.head())

# display(test_full_data.head())
outcomes = train_full_data['Survived']

#display(output)

train_data = train_full_data.drop('Survived',axis=1)

test_data = test_full_data.drop('Cabin',axis=1)

display(train_data.head())

display(test_data.head())





def accuracy_score(truth, pred):

    """ Returns accuracy score for input truth and predictions. """

    

    # Ensure that the number of predictions matches number of outcomes

    if len(truth) == len(pred): 

        

        # Calculate and return the accuracy as a percent

#         print "HEre is the calculation \n",format((truth == pred))

        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)

    

    else:

        return "Number of predictions does not match number of outcomes!"

vs.survival_stats(train_data, outcomes, 'Sex')







vs.survival_stats(test_data, outcomes, 'Pclass', ["Sex == 'male'", "Age < 10"])





vs.survival_stats(test_data, outcomes, 'Pclass', ["Sex == 'female'", "Age > 40"])



vs.survival_stats(test_data, outcomes, 'Age', ["Sex == 'female'", "Pclass == 1"])



def predictions_3(data):

    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    

    predictions = []

    for _, passenger in test_data.iterrows():

        

        # Remove the 'pass' statement below 

        # and write your prediction conditions here

        

        if passenger['Sex'] == 'male' and passenger['Age'] < 10 and passenger['Pclass'] != 3:

            predictions.append(1)

        elif passenger['Sex'] == 'female':

            if passenger['Pclass'] == 3 and passenger['Age'] > 40:

                predictions.append(0) 

            else:

                predictions.append(1)

        else:

            predictions.append(0)

        

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_3(data)

# print(predictions)
df = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})

print(df)

df.to_csv('submission1.csv', encoding='utf-8')




