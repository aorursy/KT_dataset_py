import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split
import math

pd.set_option("display.max_columns", None)
os.listdir("../input/uncover/UNCOVER/covid_tracking_project/")
"""
def load_data():
  # Run this cell to download our data into a file called 'cifar_data'
  import gdown
  gdown.download('https://drive.google.com/uc?id=1-BjeqccJdLiBA6PnNinmXSQ6w5BluLem','cifar_data','True'); # dogs v road;

  # now load the data from our cloud computer
  import pickle
  data_dict = pickle.load(open( "cifar_data", "rb" ));
  
  data   = data_dict['data']
  labels = data_dict['labels']
  
  return data, labels
"""
DATA_DIR = "../input/uncover/UNCOVER/covid_tracking_project/"
us_covid_data = pd.read_csv(DATA_DIR + "covid-statistics-by-us-states-daily-updates.csv")
# how many rows do we have?
us_covid_data.shape
# ANSWER: one state's cases, for one day
us_covid_data.iloc[0, :]
# get basic summary statistics
us_covid_data.describe()
# get number of NaNs
us_covid_data.isna().sum()
filtered_us_covid_data = us_covid_data[["date", "state", "positive", "negative", "death", "total", "totaltestresults", 
                                        "deathincrease", "hospitalizedincrease", "negativeincrease", "positiveincrease", 
                                        "totaltestresultsincrease"]]
# check out the number of positive test cases, across all states, by day since April 28th
us_covid_data["positive"].hist()
plt.show()
# check out negative cases
us_covid_data["negative"].hist()
plt.show()

# check out, by state
state = input("Insert the abbreviation for the state:")
print("\n")

if state not in filtered_us_covid_data["state"].unique():
    raise ValueError("Abbreviation isn't a state. Please try again")

print(f"You've chosen to see the data for the state of {state}")

# check out positive graphs
us_covid_data[us_covid_data["state"] == state]["positive"].hist()
plt.title(f"Graph of positive COVID cases, from January 22nd to April 28th, for the state of {state}")
plt.xlabel("Number of cases, on a given day")
plt.ylabel("Number of days with that many cases")
plt.show()
# convert date column to dates (from string)
us_covid_data["converted_date"] = pd.to_datetime(us_covid_data["date"], format="%Y-%m-%d")
# set date as the index
us_covid_data.set_index("converted_date", inplace = True)
us_covid_data
us_covid_data[us_covid_data["state"] == state]["positive"]
us_covid_data["week_of_year"] = pd.Int64Index(us_covid_data.index.isocalendar().week)
# check out, by state
state = input("Insert the abbreviation for the state:")
print("\n")

if state not in filtered_us_covid_data["state"].unique():
    raise ValueError("Abbreviation isn't a state. Please try again")

print(f"You've chosen to see the data for the state of {state}")

# check out positive graphs
us_covid_data[us_covid_data["state"] == state][["positive", "week_of_year"]].plot(x = "week_of_year", y = "positive")
plt.title(f"Number of positive cases, by week, for the state of {state}")
plt.ylabel("Number of positive cases")
plt.xlabel("Week of the year")
plt.show()
# create X, y training sets
X = us_covid_data[["state","positive", "negative","total", "totaltestresults", "deathincrease", "week_of_year"]]
y = us_covid_data["positiveincrease"]

# let's set the index to be the state
X.set_index("state", inplace=True)

# let's impute any NaNs with 0
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# first model: linear regression
from sklearn import linear_model

# set up our model
linear = linear_model.LinearRegression(fit_intercept = True)

# train the model 
linear.fit(X_train, y_train)

# make predictions
y_pred = linear.predict(X_test)
# evaluate performance with MSE
from sklearn.metrics import mean_squared_error

# calculate MSE
mse_linear = mean_squared_error(y_true = y_test, 
                                y_pred = y_pred)

# get RMSE
rmse_linear = math.sqrt(mse_linear / X_test.shape[0])

print(f"Our RMSE (root mean squared error) for the linear regression is: {rmse_linear}")
from sklearn.neural_network import MLPClassifier
# Create and train our multi layer perceptron model
nnet = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter= 100)  ## How many hidden layers? How many neurons does this have?
nnet.fit(X_train, y_train)

# Predict what the classes are based on the testing data
nnet_preds = nnet.predict(X_test)

# calculate MSE
mse_nnet = mean_squared_error(y_true = y_test, 
                              y_pred = nnet_preds)

# get RMSE
rmse_nnet = math.sqrt(mse_nnet / X_test.shape[0])

print(f"Our RMSE (root mean squared error) for the neural network is: {rmse_nnet}")
for ilayer in [(1,1), (3,3), (5,5), (8,6), (10,10,10), (10,10,5)]:
    
    # get shape of hidden layers
    print(f"Shape of hidden layers: {ilayer}")

    # fit neural net
    nnet = MLPClassifier(hidden_layer_sizes=ilayer, max_iter= 1000)  ## How many hidden layers? How many neurons does this have?
    nnet.fit(X_train, y_train)

    # Predict what the classes are based on the testing data
    nnet_preds = nnet.predict(X_test)

    # calculate MSE
    mse_nnet = mean_squared_error(y_true = y_test, 
                                  y_pred = nnet_preds)

    # get RMSE
    rmse_nnet = math.sqrt(mse_nnet / X_test.shape[0])

    # print results
    print(f"Our parameters are: {size} layers, {numNeurons} neurons, and {numIt} iterations\nOur RMSE (root mean squared error) for the neural network on the test set is: {rmse_nnet}")
    print("=====================")
