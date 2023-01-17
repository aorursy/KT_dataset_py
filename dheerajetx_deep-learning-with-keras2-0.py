#This notebook contains an example of creating a neural network in Keras2.0. 
#Dataset used for this model is of a 'Game software' in which the various features determine the selling price per unit of the software. 
#Our goal is create, train a model and finally determine the selling price of a new Game software for the data containg the values of features(on which model was trained from training dataset).
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
# %load ../input/create_model final.py
import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("../input/sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")
print("Done")
from sklearn.preprocessing import MinMaxScaler

training_data_df = pd.read_csv("../input/sales_data_training.csv")
training_data_df.head()
test_data_df = pd.read_csv("../input/sales_data_test.csv")
test_data_df.head()
#Data needs to be scaled to a small range like 0 to 1 for the neural network network to work well. 
scaler = MinMaxScaler(feature_range=(0,1))
#scaling both the training inputs and outputs 
scaled_taining = scaler.fit_transform(training_data_df)
scaled_testing = scaler.fit_transform(test_data_df)


type(scaled_testing)

#Import keras 

from keras.models import Sequential
from keras.layers import *
training_data_df = pd.read_csv("../input/sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values

Y= training_data_df[['total_earnings']].values
#Define the Model 

model = Sequential()
#Define the Model cntd..
model.add(Dense(50, input_dim= 9, activation='relu'))          #There are 9 features in the training data.

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

#output layer

model.add(Dense(1, activation='linear'))

#final step of building a model is to compile it.
model.compile(loss="mean_squared_error", optimizer="adam")

#Train the model
model.fit(
X,
Y,
epochs= 50,     #epoch: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
shuffle= True,  #NN works better on shuffled data
verbose= 2)     #verbose=2 for more detailed output

#Load the separate test data set.
test_data_df = pd.read_csv("../input/sales_data_testing_scaled.csv")
X_test = test_data_df.drop('total_earnings', axis = 1 ).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test,Y_test,verbose= 0)

print("The mean squared error (MSE) for the test data is {}". format (test_error_rate))
#Prediction- Load the data we make to use a make a prediction 

X = pd.read_csv("../input/proposed_new_product.csv")

#make a prediction with the neural network

prediction = model.predict(X)

#Grab the first element of first prediction(here only one prediction is present)

prediction[0][0]
#Re-scale the data from 0 to 1 range back to dollars

#These constants are from when the data was oroginally scaled down to the 0-to-1 

prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earning Predictions for Proposed Product- ${}". format(prediction[0][0]))

#Now save(to disk) the trained model so that every time we will not need to re-run it for training the model. 

model.save("trained_model.h5")  # It will save the structure of NN and trained weight coefficients to h5 format.
#h5 format is binary file format for storing Python array data.

print("Model saved to the disk")
import os
#os.getcwd()
os.chdir('/kaggle/working')
os.listdir()

#Load the saved model

from keras.models import load_model

model = load_model("trained_model.h5")

#Predict the value with this loaded model
X = pd.read_csv("../input/proposed_new_product.csv")

#make a prediction with the neural network

prediction = model.predict(X)

#Grab the first element of first prediction(here only one prediction is present)

prediction[0][0]
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earning Predictions for Proposed Product- ${}". format(prediction[0][0]))
