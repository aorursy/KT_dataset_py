# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 # Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from IPython.display import display # Allows the use of display() for DataFrames

 
# Pretty display for notebooks
import matplotlib.pyplot as plt
 
df = pd.read_csv('../input/train.csv')
df.head()
# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = df['Survived']
#print(outcomes)

#Having dropped the 'Survived class' now I have kind of data without the final information if they died or not
data = df.drop('Survived', axis = 1)
display(data.head())

# Show the new dataset with 'Survived' removed
#display(data.head())
def accuracy(truth, pred):
    """ I will measure our predictions against the truth, trying to understand how accurate was our model """
    
    # We need to make sure that the number of predictions is the same of the data that I have available
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Wrong match between predictions and outcomes, let's try again"
    
def predictions(data): 
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            if passenger['Sex'] == 'male' and passenger['Age'] < 16 and (passenger['Pclass'] == 1 or passenger['Pclass'] == 2):  
             predictions.append(1) 
            
            
            else:
             predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)
 
# Make the predictions
predictions = predictions(data)
print(accuracy_score(outcomes, predictions))
df['Age'].plot(kind ="hist", bins = 10)
plt.xlabel("Age")
#this is the distribution of the passengers of Titanic, according to their age
#df.hist()
df.groupby('Survived').plas.hist()
 
 