# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print(train.columns)

print(train.shape)

train.head(10)
train.describe()
# SQFT_LIVING should not be null as we are trying to predict that, 

# Ensuring we are sending the labelled data for training rather than incomplete or missing labels

null = pd.isnull(train["sqft_living"])

train[null]
# Ensuring all columns have no none or null values.

missing_values = train.isnull()

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#Prepare input X parameters/features and output y



# Split data into 'X' features and 'y' target label sets

X = train[['price','bedrooms', 'bathrooms']]

y = train['sqft_living']
# Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=100)
print(X_train.shape)



print(y_test.shape)
# Run the model



# Import model for fitting

from sklearn.linear_model import LinearRegression



#You can try follwoing variation on above model, above is just default one

model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)



output_model
#Save the model in pickle

#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

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

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))