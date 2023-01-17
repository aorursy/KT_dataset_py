# Import Library

# Import data analysis modules

import numpy as np

import pandas as pd

import os

# To save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.chdir(r'/kaggle/input/housesalesprediction')

train = pd.read_csv('kc_house_data.csv')

train.head(5)
train.describe()
train.dtypes
# Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.head(5)
# Visualize the data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
# Split data into 'X' features and 'y' target label sets

X = train[['bedrooms']]

y = train['sqft_living']
# Split data into Train and test

# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
print(X_train.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)

#output =X_test

#output['vehicleTypeId'] = y_test

output_model
#Save the model in pickle

#Save to file in the current working directory

pkl_filename = "/kaggle/working/pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
# Understanding accuracy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(X_test)

#print("",classification_report(y_test, predictions))

#print("confusion_matrix",confusion_matrix(y_test, predictions))

#print("accuracy_score",accuracy_score(y_test, predictions))

##**Accuracy is a classification metric. You can't use it with a regression. See the documentation for info on the various metrics.

#For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).

#print("Score",score(y_test, X_test))

#score(self, X, y, sample_weight=None)

## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show()
# Finding MAE, MSE and RMSE

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))