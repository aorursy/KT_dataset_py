# importing packages
import numpy as np # to perform calculations 
import pandas as pd # to read data
import matplotlib.pyplot as plt # to visualise
# In read_csv() function, we have passed the location to where the file is located at dphi official github page
boston_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Boston_Housing/Training_set_boston.csv")
boston_data.head()
X = boston_data.drop('MEDV', axis = 1)    # Input Variables/features
y = boston_data.MEDV      # output variables/features
# import train_test_split
from sklearn.model_selection import train_test_split 

# Assign variables to capture train test split output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train: independent/input feature data for training the model
# y_train: dependent/output feature data for training the model
# X_test: independent/input feature data for testing the model; will be used to predict the output values
# y_test: original dependent/output values of X_test; We will compare this values with our predicted values to check the performance of our built model.
 
# test_size = 0.20: 20% of the data will go for test set and 70% of the data will go for train set
# random_state = 42: this will fix the split i.e. there will be same split for each time you run the code
# find the number of input features
n_features = X.shape[1]
print(n_features)
from tensorflow.keras import Sequential    # import Sequential from tensorflow.keras
from tensorflow.keras.layers import Dense  # import Dense from tensorflow.keras.layers
from numpy.random import seed     # seed helps you to fix the randomness in the neural network.  
import tensorflow
# define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
# import RMSprop optimizer
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop(0.01)    # 0.01 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model
seed_value = 42
seed(seed_value)        # If you build the model with given parameters, set_random_seed will help you produce the same result on multiple execution


# Recommended by Keras -------------------------------------------------------------------------------------
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# Recommended by Keras -------------------------------------------------------------------------------------


# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tensorflow.random.set_seed(seed_value) 
model.fit(X_train, y_train, epochs=200, batch_size=30, verbose = 1)
model.evaluate(X_test, y_test)
####################### Complete example to check the performance of the model with different learning rates #######################################
# define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

optimizer = RMSprop(0.1)    # 0.1 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model

# fit the model 
model.fit(X_train, y_train, epochs=200, batch_size=30, verbose = 1)

# evaluate the model
print('The MSE value is: ', model.evaluate(X_test, y_test))
####################### Complete example to check the performance of the model with different epochs and learning rate = 0.01 #######################################
# define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

optimizer = RMSprop(0.01)    # 0.1 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model

# fit the model 
model.fit(X_train, y_train, epochs=10, batch_size=30, verbose = 1)

# evaluate the model
print('The MSE value is: ', model.evaluate(X_test, y_test))
####################### Complete example to check the performance of the model with different batch size while keeping epochs as 30 and learning rate as 0.01 #######################################
# define the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

optimizer = RMSprop(0.01)    # 0.1 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model

# fit the model 
model.fit(X_train, y_train, epochs=200, batch_size=40, verbose = 1)

# evaluate the model
print('The MSE value is: ', model.evaluate(X_test, y_test))
# Import the GridSearchCV class
from sklearn.model_selection import GridSearchCV

# 1. Define the model's architecture
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
optimizer = RMSprop(0.1)    # 0.1 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model

# 2. Define the hyperparameters grid to be validated
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)

# 3. Run the GridSearchCV process
grid_result = grid.fit(X_train, y_train)

# 4. Print the results of the best model
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# ----------------------------- Functional Tuning - Option 1: using Sklearn  ------------------------------
# Goal: tune the batch size and epochs

# Import KerasRegressor class
from keras.wrappers.scikit_learn import KerasRegressor

# Define the model trhough a user-defined function
def create_model(optimizer=RMSprop(0.01)):
  model = Sequential()
  model.add(Dense(10, activation='relu', input_shape=(n_features,)))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)    # compile the model
  return model
model = KerasRegressor(build_fn=create_model, verbose=1)

# Define the hyperparameters grid to be validated
batch_size = [10, 20, 30, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
model = KerasRegressor(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

# Run the GridSearchCV process
grid_result = grid.fit(X_train, y_train, verbose = 1)

# Print the results of the best model
print('Best params: ' + str(grid_result.best_params_))
print('Best params: ' + str(grid_result.best_params_))
# Import the cross validation evaluator
from sklearn.model_selection import cross_val_score

# Measure the model's performance
results = cross_val_score(grid.best_estimator_, X_test, y_test, cv=5)
print('Results: \n  * Mean:', -results.mean(), '\n  * Std:', results.std())
print('Results: \n  * Mean:', -results.mean(), '\n  * Std:', results.std())
# ----------------------------- Functional Tuning - Option 2: using Keras Tuner ------------------------------
# Goal: tune the learning rate

# 0. Install and import all the packages needed
!pip install -q -U keras-tuner
import kerastuner as kt

# 1. Define the general architecture of the model through a creation user-defined function
def model_builder(hp):
  model = Sequential()
  model.add(Dense(10, activation='relu', input_shape=(n_features,)))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1))
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4]) # Tuning the learning rate (four different values to test: 0.1, 0.01, 0.001, 0.0001)
  optimizer = RMSprop(learning_rate = hp_learning_rate)                            # Defining the optimizer
  model.compile(loss='mse',metrics=['mse'], optimizer=optimizer)                   # Compiling the model 
  return model                                                                     # Returning the defined model

# 2. Define the hyperparameters grid to be validated
tuner_rs = kt.RandomSearch(
              model_builder,                # Takes hyperparameters (hp) and returns a Model instance
              objective = 'mse',            # Name of model metric to minimize or maximize
              seed = 42,                    # Random seed for replication purposes
              max_trials = 5,               # Total number of trials (model configurations) to test at most. Note that the oracle may interrupt the search before max_trial models have been tested.
              directory='random_search')    # Path to the working directory (relative).

# 3. Run the GridSearchCV process
tuner_rs.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
# 4.1. Print the summary results of the hyperparameter tuning procedure
tuner_rs.results_summary()
# 4.2. Print the results of the best model
best_model = tuner_rs.get_best_models(num_models=1)[0]
best_model.evaluate(X_test, y_test)
# 4.3. Print the best model's architecture
best_model.summary()
# Load new test data
new_test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Boston_Housing/Testing_set_boston.csv')
# make a prediction
best_model.predict(new_test_data)