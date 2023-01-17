# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





from IPython.display import display # Allows the use of display() for DataFrames



# Import supplementary visualizations code visuals.py





# Pretty display for notebooks

%matplotlib inline



# Load the dataset

in_file = '../input/train.csv'

full_data = pd.read_csv(in_file)



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

print (accuracy_score(outcomes[:5], predictions))
def predictions_3(data):

    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    

    predictions = []

    for _, passenger in data.iterrows():

        

        # Remove the 'pass' statement below 

        # and write your prediction conditions here

        if passenger['Sex']=='female':

            if passenger['Pclass']==3:

                if passenger['SibSp']>0 and passenger['Parch']>0:

                    predictions.append(0)

                else:

                    predictions.append(1)

            else:

                predictions.append(1)

        else:

            if passenger['Sex']=='male'and passenger['Age']<10:

                predictions.append(1)

            elif passenger['Sex']=='male'and passenger['Embarked']=='S'and passenger['Pclass']==3:

                predictions.append(0)

            else:

                predictions.append(0)



        

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_3(data)

print (accuracy_score(outcomes, predictions))