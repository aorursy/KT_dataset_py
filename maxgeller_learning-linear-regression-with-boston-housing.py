# import numpy and pandas

# also import warnings and ignore them to keep notebook clean

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore") # ignores warnings

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# loading in the dataset and peeking at the first five rows

data = pd.read_csv('../input/BostonHousing.csv')

data.head()
# Taking some time here to play with some Pandas methods

#data.index # shows range of indices

#data.to_numpy() # returns df as a matrix

#data.describe() # some summary statistics. Notice the counts are all equal so there are no missing values--clean

#data.info() # gives us some more information and count and datatypes

#data.T # switches columns with rows

#data.sort_index(axis=1, ascending = False)# sorts the dataset by the index of each row

#data["tax"] # returns a single column (series)

#data[0:3] # row slicing

#data["tax"][3] # locate a specific value in column. Also can be done with data['tax'].loc[3]

#data.loc[:, ["tax","nox"]] # returns all the rows of two features

#data.isnull().sum() the first checks for missing values, the second sums them up
# Create empty list for coefficients

coefficients = []
# Creating helper functions to make model more viewable

def reshape_X(X):

    return X.reshape(-1,1) # numpy.reshape returns the m x n matrix of the arguments in this case
# The second helper matrix concatenates a feature of ones to the matrix

def concatenate_ones(X):

    ones = np.ones(shape=X.shape[0]).reshape(-1,1) # np.ones() creates an array of ones

    return np.concatenate((ones, X), 1) # concatenate basically appends the newly created vector of ones
# creating our function to fit the training data

def fit(X,y):

    global coefficients

    if len(X.shape) == 1:

        X = reshape_X(X)

    X = concatenate_ones(X)

    coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y) # math to obtain coeff ie. slope

    print(coefficients)
# creating a predict function to predict coefficient(??)

def predict(entry):

    b0 = coefficients[0] #initial slope

    other_betas = coefficients[1:] 

    prediction = b0 # initial prediction

    

    for xi, bi in zip(entry, other_betas): 

        # we avoid declaring two for loops by assigning xi to entry and bi to coef.

        # zip function creates a tuple out of the entry and other_betas

        prediction += (bi * xi)

    return prediction

        
X = data.drop("medv", axis=1).values # drops the medv column from the data

y = data["medv"].values # setting our target equal to the values we just dropped
fit(X,y) # fits our dataset with the model
predict(X[0])
predictions = []

for row in X:

    predictions.append(predict(row))
results = pd.DataFrame({

    "Actual": y,

    "Predicted": predictions

})
# importing matplotlib for graphs 

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

linear_regressor.fit(X,y)

Y_pred = linear_regressor.predict(X)
plt.scatter(predictions, y)

plt.plot(predictions, Y_pred, color='red')

plt.show()