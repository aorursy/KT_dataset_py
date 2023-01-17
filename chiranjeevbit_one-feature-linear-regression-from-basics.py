# Import the required libraries

from sklearn import linear_model

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math   # yep! going to a bit of maths later!!

from scipy import stats as st # and some stats
Xtrain = pd.read_csv("../input/Linear_X_Train.csv")

Ytrain = pd.read_csv("../input/Linear_Y_Train.csv")

Xtest = pd.read_csv("../input/Linear_X_Test.csv")
print (Ytrain.loc[0:5])

print (Xtrain.loc[0:5])
print(Xtrain.isnull().sum())

print(Ytrain.isnull().sum())

print(Xtest.isnull().sum())

# Review some of the statistics to check whether the data is skewed

print ("Mean of X Training set: ", np.mean(Xtrain), "\n")

print ("Median of X Training set: ", np.median(Xtrain), "\n")

print ("Mean of Y Training set: ", np.mean(Ytrain), "\n")

print ("Median of Y Training set: ", np.median(Ytrain), "\n")

print ("Std Dev of X Training set: ", np.std(Xtrain), "\n")

print ("Std Dev of Y Training set: ", np.std(Xtrain), "\n")
### Now plot the relation between X and Y

### this gives some core routines to experiment with different parameters

plt.title('Relationship between X and Y')

plt.xlabel("X label ")

plt.ylabel("Y label ")

plt.scatter(Xtrain, Ytrain,  color='red')

plt.show()





# Create linear regression object

lm = linear_model.LinearRegression()



# then fit it

lm.fit(Xtrain,Ytrain)



# R sq to give an idea of the fit 

print('R sq: ',lm.score(Xtrain,Ytrain))



# and so the correlation is..

print('Correlation: ', math.sqrt(lm.score(Xtrain,Ytrain)))
# This the coefficient for the single feature

print("Coefficient for X ", lm.coef_)



# Get the standard error

print ("Standard Error: ",st.sem(Xtrain))

# Our hypothesis test for this is that there is no difference between the coefficient and the true value

# that the coefficient should be

ttest = lm.coef_/st.sem(Xtest)

print ("The t-statistic:",ttest)



# but we need the p-value to help determine the probablity that we have a correct t-statistic

print ("Two tailed p-values:")

st.pearsonr(Xtrain, Ytrain)
y_predicted = lm.predict(Xtest)
y_predicted
np.savetxt('predictedYtext.csv',y_predicted)