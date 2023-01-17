import numpy as np

import pandas as pd



# RMS Titanic data visualization code 

from IPython.display import display

%matplotlib inline



# Load the dataset

in_file_train = '../input/train.csv'

in_file_test = '../input/test.csv'



full_data = pd.read_csv(in_file_train)

test_df = pd.read_csv(in_file_test)



# Print the first few entries of the RMS Titanic data

display(full_data.head())
# Store the 'Survived' feature in a new variable and remove it from the dataset

outcomes = full_data['Survived']

data = full_data.drop('Survived', axis = 1)



# Show the new dataset with 'Survived' removed

display(data.head())
def accuracy_score(truth, pred):

    """ Returns accuracy score for input truth and predictions. """

    

    # Ensure that the number of predictions matches number of outcomes

    if len(truth) == len(pred): 

        

        # Calculate and return the accuracy as a percent

        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)

    

    else:

        return "Number of predictions does not match number of outcomes!"

    

# Test the 'accuracy_score' function

predictions = pd.Series(np.ones(5, dtype = int))

accuracy_score(outcomes[:5], predictions)
def predictions_0(data):

    """ Model with no features. Always predicts a passenger did not survive. """



    predictions = []

    for _, passenger in data.iterrows():

        

        # Predict the survival of 'passenger'

        predictions.append(0)

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_0(data)
accuracy_score(outcomes, predictions)
survival_stats(data, outcomes, 'Sex')
def predictions_1(data):

    """ Model with one feature: 

            - Predict a passenger survived if they are female. """

    

    predictions = []

    for _, passenger in data.iterrows():

        

        # Remove the 'pass' statement below 

        # and write your prediction conditions here

        if passenger['Sex'] == 'female':

            predictions.append(1)

        else:

            predictions.append(0)

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_1(data)
accuracy_score(outcomes, predictions)
survival_stats(data, outcomes, 'Pclass', ["Sex == 'male'"])
def predictions_2(data):

    """ Model with two features: 

            - Predict a passenger survived if they are female.

            - Predict a passenger survived if they are male and younger than 10. """

    

    predictions = []

    for _, passenger in data.iterrows():

        

        # Remove the 'pass' statement below 

        # and write your prediction conditions here

        if passenger['Sex'] == 'female':

            predictions.append(1)

        else:

            if passenger['Age'] < 10:

                predictions.append(1)

            else:

                predictions.append(0)

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_2(data)
accuracy_score(outcomes, predictions)
survival_stats(data, outcomes, 'Parch', ["Sex == 'male'", "Embarked == 'S'", "Pclass == 1"])
def predictions_3(data):

    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    

    predictions = []

    for _, passenger in data.iterrows():

        

        # Remove the 'pass' statement below 

        # and write your prediction conditions here

        if passenger['Sex'] == 'female':

            predictions.append(1)

        else:

            if passenger['Age'] < 10:

                predictions.append(1)

            else:

                if passenger['Pclass'] == 1:

                    if passenger['Embarked'] == 'S' and passenger['Parch'] == 2:

                        predictions.append(1)

                    else:

                        if passenger['Age'] > 20 and passenger['Age'] < 40:

                            if passenger['SibSp'] == 0 and passenger['Parch'] == 0:

                                if passenger['Fare'] >= 20 and passenger['Fare'] <= 40:

                                    predictions.append(1)

                                else:

                                    predictions.append(0)

                            else:

                                predictions.append(0)

                        else:

                            predictions.append(0)

                else:

                    predictions.append(0)

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_3(data)
accuracy_score(outcomes, predictions)
predictions = predictions_3(test_df)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('titanic.csv', index=False)