# Import all the necessary required modules



import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import seaborn as sb

from scipy import stats

import numpy as np
# Import the 'mtcars' dataset --> mtcars.csv



data = pd.read_csv('../input/mtcars/mtcars.csv')
# Displays the first five observations in mtcars dataset



data.head()
# Fetches the count of rows and columns in mtcars dataset



data.shape
# Linear Regression is used to predict the 'mpg'(Mileage) == y

# Using the predictors like 'hp'(Horse Power), 'wt'(Weight) == X



# Checking for Normality Test for all the predictors

print("Range of SKEWNESS and KURTOSIS to be between -1 and +1 for all the Predictors")



# Measure of Skewness

print("Skewness of 'hp': ", stats.skew(data.hp))

print("Skewness of 'wt': ", stats.skew(data.wt))



# Measure of Kurtosis

print("Kurtosis of 'hp': ", stats.kurtosis(data.hp))

print("Kurtosis of 'wt': ", stats.kurtosis(data.wt))
# Visually respresenting this for 'hp'

%matplotlib inline

sb.distplot(data.hp);
# Visually respresenting this for 'wt'

%matplotlib inline

sb.distplot(data.wt);
# Declaring the Predictors and Target variables



X = data.loc[:,['hp', 'wt']]

y = data.mpg
# Splitting the dataset into Train and Test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=14)
# Verify the shape(number of records) after split



print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)



print("y_train shape:", y_train.shape)

print("y_train shape:", y_test.shape)
# Defining the Linear Regression Model

model = LinearRegression()
# By passing the X, y train data fit the model

model.fit(X_train, y_train)
# Predict y (mpg) by passing the X_test

y_predict = model.predict(X_test)
# Estimating the accuracy of the model using 'r2 score'

r2_score(y_test, y_predict)*100
# Comparing the y_test (test set data) with that of the data predicted by model y_predict

y_test.head(), y_predict[0:5]
# Check for Outliers

data.hp.plot(kind='box');
# Find out the maximum value causing it to be outlied

print("Max 'hp':", data.hp.max())



data[data.hp == data.hp.max()]
# Shape of dataset before outlier removal

data.shape
# Dropping the outlier record from the dataset

data_without_outlier = data.drop(index=30)
# Shape of dataset after outlier removal - new updated dataset named 'data_without_outlier'

data_without_outlier.shape
# Run the entire model again with new updated dataset after removal of the outlier



# Declaring the Predictors and Targets

X = data_without_outlier.loc[:,['hp', 'wt']]

y = data_without_outlier.mpg



# Split the new updated dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=14)



# Defining the Model

model = LinearRegression()



# Fitting the Model

model.fit(X_train, y_train)



# Perform prediction

y_predict = model.predict(X_test)



# Measure the accuracy

r2_score(y_test, y_predict)*100
y_test
y_predict
model.predict([[110, 3.2]])