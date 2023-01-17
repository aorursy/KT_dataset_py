# Import the required libraries

from sklearn import linear_model

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math 

from scipy import stats as st # and some stats



# Load up the files 

dirty_training_set = pd.read_csv('../input/train.csv')

dirty_test_set = pd.read_csv('../input/test.csv')



# Clean the data by dropping NA values

training_set = dirty_training_set.dropna() 

test_set = dirty_test_set.dropna() 



# Convert all data to matrix for easy consumption

x_training_set = training_set.as_matrix(['x'])

y_training_set = training_set.as_matrix(['y'])

x_test_set = test_set.as_matrix(['x'])

y_test_set = test_set.as_matrix(['y'])





# So let's plot some of the data 

# - this gives some core routines to experiment with different parameters

plt.title('Relationship between X and Y')

plt.scatter(x_training_set, y_training_set,  color='black')

plt.show()





# So let's plot some of the data 

# - this gives some core routines to experiment with different parameters

plt.title('Relationship between X and Y')

plt.scatter(x_test_set, y_test_set,  color='black')

plt.show()



# Now to set up the linear regression model

# Create linear regression object

lm = linear_model.LinearRegression()

# ... then fit it

lm.fit(x_training_set,y_training_set)



# Have a look at R sq to give an idea of the fit 

print('R sq: ',lm.score(x_training_set,y_training_set))



# and so the correlation is..

print('Correlation: ', math.sqrt(lm.score(x_training_set,y_training_set)))



# Equation coefficient and Intercept

print("Coefficient for X: ", lm.coef_)



print("Intercept for X: ", lm.intercept_)



# So let's run the model against the test data

y_predicted = lm.predict(x_test_set)



plt.title('Comparison of Y values in test and the Predicted values')

plt.ylabel('Test Set')

plt.xlabel('Predicted values')

plt.plot(y_predicted, '.', y_test_set, 'x')

plt.show()
