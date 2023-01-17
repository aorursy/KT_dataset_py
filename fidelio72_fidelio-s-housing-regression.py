# https://www.youtube.com/watch?v=3S9j71OL1H0 

# Load Libraries

# Data Objects
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

# Data Visualization
from matplotlib import pyplot as plt
from pylab import rcParams

# Machine Learning Algorythms and scoring
from scipy import stats
from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Lasso # This is the linear regression algo that uses an L1 regularization that pushes weights down to 0 - making feature selection easier. 

#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load Data
trainingData = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
competitionData = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Set Plotting parameters
%matplotlib inline
rcParams['figure.figsize'] = 10,4
print("Training Data Shape (Rows,Columns):",trainingData.shape)
print("Competition Data Shape (Rows,Columns):", competitionData.shape)
display(trainingData.head())
display(competitionData.head())
# Lets see all the columns
pd.options.display.max_columns = None
display(trainingData.head())
display(competitionData.head())
trainingDataColumns = list(trainingData)  # For discrete items, replace nan with "Missing", for continuous items filled with 0

for c in trainingDataColumns:
    if trainingData[c].dtype == 'O':
        trainingData[c].fillna(value = 'Missing', inplace = True) #all nan values replaced with value i'm going to give it.. which is Missing. In place means we're changing that data object value 
    else:
        trainingData[c].fillna(0, inplace = True)
        

# Do the same for competition data

competitionDataColumns = list(competitionData)

for f in competitionDataColumns:
    if competitionData[f].dtype == 'O':
        competitionData[f].fillna(value = 'Missing', inplace = True) #all nan values replaced with value i'm going to give it.. which is Missing. In place means we're changing that data object value 
    else:
        competitionData[f].fillna(0, inplace = True)
        
display(trainingData.head())
display(competitionData.head())

# Transform discrete values to columns with 1 and 0s
trainingData = pd.get_dummies(trainingData) #tranform the unique values and create new features 1 or 0, allowing us to do the linear regression --> now continuous vlaues

competitionData = pd.get_dummies(competitionData) 

display(trainingData.head())
display(competitionData.head())

print("Training Data Shape (Rows,Columns):",trainingData.shape)
print("Competition Data Shape (Rows,Columns):",competitionData.shape)
#Training data and competitive data need to have the same number of features (Columns)
#Let's drop the features that don't line up! Make those dataframes exactly the same. 
#OOOps! There is a difference between features in the training data set and the competition dat set

sp = trainingData['SalePrice']

missingFeatures = list(set(trainingData.columns.values) - set(competitionData.columns.values))
trainingData = trainingData.drop(missingFeatures,axis=1)

missingFeatures = list(set(competitionData.columns.values) - set(trainingData.columns.values))
competitionData = competitionData.drop(missingFeatures,axis=1)

print("Training Data Shape (Rows,Columns):",trainingData.shape)
print("Competition Data Shape (Rows,Columns):",competitionData.shape)



# Give it a random state (being 0) - the random 75/25 split will be 
# It returns 4 different objects, and these are defining those items. 

x_train, x_test, y_train, y_test = train_test_split(trainingData, sp, random_state=0)
# Lasso is a form of linear regression that restricts coefficients to be close to zero or exactly zero 
# This acts as a form of automatic feature selection
# alpha is how strongly the coefficients are pushed to zero
# perform a loop on alpha to get the one that returned the highest test scores, removed for faster performance


# - Instantiate an instance - this is now an instance of the Lasso model - called myModel
# Parameter of 298.4 was -- did a loop to loop through all the alphas that gave me the best test score... sounds pretty fun and complicated
# the .fit method determines those weights. x_train and y_train are (x are the features, y are the prices) - uses gradient decents - uses L1 to push those done

myModel = Lasso(alpha = 298.4).fit(x_train,y_train) 


print ("Train score: ", myModel.score(x_train,y_train),"\nTest score: ", myModel.score(x_test,y_test))
print ("Number of features used: {}".format(np.sum(myModel.coef_ != 0))) #All of the weights that were not 0, what were the features considered. 

submission = pd.DataFrame(myModel.predict(competitionData), columns=['SalePrice'], index = competitionData.index)

display(submission.head())
submission.to_csv("submission_bhart.csv")
#Examine Top Ten Features

featureDF = pd.DataFrame(trainingData.columns.values,columns=['Features'])
featureDF['w'] = myModel.coef_
featureDF['Abs(w)'] = featureDF['w'].abs()
featureDF = featureDF.sort_values(by=['Abs(w)'], ascending=False)
featuresUsed = featureDF.head(10)

display(featuresUsed[['Features','w']])