# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np                                      # linear algebra

import pandas as pd                                     # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt                         # plotting

from sklearn.linear_model import LinearRegression       # linear regression

from sklearn.model_selection import train_test_split    # can use this for splitting the data into training and testing sets

from sklearn.metrics import mean_squared_error          # mean squared error metric



# Any results you write to the current directory are saved as output.



# Read in the train & test data csv:

train_and_test = pd.read_csv('/kaggle/input/relex-beer-challenge/cerveja_train_and_test_2.csv', parse_dates = ['Data'])

# Divide the data into separate train and test sets:

training_data, testing_data = train_test_split(train_and_test, test_size=0.2)



# TRAINING BEGINS

# Take average temperatures from the training set:

x = training_data['Temperatura Media (C)']

# Take the beer consumption (aka response) from the training set:

y = training_data['Consumo de cerveja (litros)']

x, y = np.array(x).reshape((-1, 1)), np.array(y)



# Fit a linear regression model with one regressor (average temperature):

model = LinearRegression().fit(x, y)

# Now you have a model that you can use to predict beer consumption!



# TESTING BEGINS

# Take average temperatures from the test set:

x_test = testing_data['Temperatura Media (C)']

x_test = np.array(x_test).reshape((-1, 1))

# Predict beer consumption using the model you trained above:

y_test_predicted = model.predict(x_test)

# Take the true, observed values of beer consumption from the test data:

y_test_ground_truth = testing_data['Consumo de cerveja (litros)']



# If you have a type of notebook that supports visualization, you can use this to plot

# the average temperature (yellow)

# the correct answer (blue) 

# and your prediction of the consumption (red).

# Note how the prediction has similar shape than average temperature.

plt.plot(

    range(0, x_test.size), x_test, 'y--',

    range(0, y_test_ground_truth.size), y_test_ground_truth, 'b-',

    range(0, y_test_predicted.size), y_test_predicted, 'r-'

)
