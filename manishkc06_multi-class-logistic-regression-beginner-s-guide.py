import numpy as np        # Fundamental package for linear algebra and multidimensional arrays

import pandas as pd       # Data analysis and manipultion tool



# To ignore warnings

import warnings

warnings.filterwarnings("ignore")
# In read_csv() function, we have passed the location to where the files are located in the UCI website. The data is separated by ';'

# so we used separator as ';' (sep = ";")

red_wine_data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# Red Wine

red_wine_data.head() 
red_wine_data.columns
# Basic statistical details about data

red_wine_data.describe()
red_wine_data.quality.value_counts().plot(kind = 'bar')
# Input/independent variables

X = red_wine_data.drop('quality', axis = 1)   # her we are droping the quality feature as this is the target and 'X' is input features, the changes are not 

                                              # made inplace as we have not used 'inplace = True'



y = red_wine_data.quality             # Output/Dependent variable
# Let's check the shapes of X and y

print("Shape: ", X.shape, "Dimension: ", X.ndim)

print("Shape: ", y.shape, "Dimension: ", y.ndim)
# import train_test_split

from sklearn.model_selection import train_test_split
# split the data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 42)



# X_train: independent/input feature data for training the model

# y_train: dependent/output feature data for training the model

# X_test: independent/input feature data for testing the model; will be used to predict the output values

# y_test: original dependent/output values of X_test; We will compare this values with our predicted values to check the performance of our built model.

 

# test_size = 0.30: 30% of the data will go for test set and 70% of the data will go for train set

# random_state = 42: this will fix the split i.e. there will be same split for each time you run the code
# import Logistic Regression from sklearn.linear_model

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
# Fit the model

log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)
y_test.values
predictions
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)