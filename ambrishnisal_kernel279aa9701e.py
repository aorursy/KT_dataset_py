# Install Library if running for the first time



#!pip3 install numpy

#!pip3 install pandas

#!pip3 install matplotlib

#!pip3 install seaborn

#import os

#import  pickle - to save my model
#Step 1: Import Library

# Import data analysis modules

import numpy as np

import pandas as pd

import os

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
#Step 2 : Data import

# Use pandas to read in csv file

#os.chdir(r'C:\Users\ambrisn\Desktop')

os.chdir('C:\\Users\\ambrisn\\Desktop')

train = pd.read_excel('kc_house_data.xlsx',encoding='Latin-1', sheet_name ="kc_house_data")

#this is just a comment

train.head(5)
train.describe()
train.dtypes
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.head(5)
#Step 4.1: Visualize the data

# Use seaborn to conduct heatmap to identify missing data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
X = train[['price']]

y = train['sqft_living']
#Step 5.2: Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
#Step 5.3: Checking file types created



print(X_train.shape)

#print(X_test.head())

#print(y_train.head())

print(y_test.shape)
#Step 6 : Run the model



# Import model for fitting

from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)

#output =X_test

#output['vehicleTypeId'] = y_test

output_model
#Step 7.0 Save the model in pickle

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
df = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df
#Step 7.1: Understanding accuracy

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
# plotting regression line

ax = plt.axes()

ax.scatter(X, y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
#Step 8: Saving output file



os.chdir('C:\\Users\\ambrisn\\Desktop')

check = pd.read_excel('kc_house_data.xlsx',encoding='Latin-1', sheet_name ="kc_house_data")

print('Importing data to solve for')

print(train.head(5))
Ypredict = pickle_model.predict(check[['price']])

output=check[['price']]

output['Y_Predicted']=Ypredict

print(output)

output.to_csv(r'C:/Users/ambrisn/Desktop/' + 'Output_data.csv')

print( 'Saved with Name: Output_data.csv')    

output.to_csv( 'Linear_Model_Output_Data.csv')