# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #create a global variable where location of file will be saved for multiple usage

    global file_path

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        file_path = os.path.join(dirname, filename)

        print(file_path)



# Any results you write to the current directory are saved as output.
import pickle

import matplotlib.pyplot as plt

import seaborn as sns
#creating dataframe to perform multiple operations in terms of Data Cleaning

df = pd.read_csv(file_path)

df
#check whether data types are compatible before starting model creation

df.dtypes
#check whether any value is null

missing_values = df.isnull()

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#As we got an even plot, meaning we do not have any missing value in data

#We will plot different graphs to identify the pattern of values which could be relatable to sqft_living

sns.pairplot(data=df, x_vars=['price','zipcode', 'yr_built'], y_vars=["sqft_living"])
#As seen above, we can corelate price with sqft_living, but zip_code and yr_built are having scattered plots

#We will plot price and sqft_living as first check

#We will define independent (X) and dependent (Y) values

X = df[['price']]

y = df['sqft_living']
# Split data into Train and test

# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)



# Import model for fitting

from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)

#output =X_test

#output['sqft_living'] = y_test

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
#Normalize the price using location of building, that is, zip code

#We will find counts of zipcodes in our data

df['zipcode'].value_counts()
#98103 has the highest contribution in above data

#We will see which zip code has highest price

df2 = df.groupby('zipcode')['price'].mean()

#Zip code having maximum value of mean price

df2.idxmax()
#We will drill down data using zipcodes 98039 and 98103

df3 = df[(df['zipcode'] == 98103) | (df['zipcode'] == 98039)]

df3
X = df3[['price']]

y = df3['sqft_living']
# Split data into Train and test

# Import module to split dataset

#from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)



# Import model for fitting

#from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)

#output =X_test

#output['sqft_living'] = y_test

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
df4 = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df4
#Understanding accuracy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(X_test)

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
print(X.shape)

print(y.shape)
# plotting regression line

#X1 = X[:,0]

ax = plt.axes()

ax.scatter(X, y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
#Saving output file

op = pd.read_csv(file_path)

print('Importing data to build final mechanism')

print(df3)
Ypredict = pickle_model.predict(df3[['price']])

output=df3[['price']]

output['Y_Predicted']=Ypredict

print(output)

output.to_csv('KC_house_data_Linear_model_output.csv', index= False)

print('Saved Output')