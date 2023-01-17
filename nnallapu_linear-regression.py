'''Simple Linear Regression  '''

# Import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Creating the dataframes for training and test datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#Clean the missing values
train_df = train_df.dropna()
test_df = test_df.dropna()


#Since the data is already split into Train and Test datasets, load the values into X_train, X_test, y_train, y_test
X_train = train_df.iloc[:,:-1].values
y_train = train_df.iloc[:,1].values
X_test = test_df.iloc[:,:-1].values
y_test = test_df.iloc[:,1].values


from sklearn.linear_model import LinearRegression
#Create an object called regressor of Linear Regression class and fit the training set to it
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Y_pred will give the predicted values of test dataset
y_pred = regressor.predict(X_test)

#Create a scatter plot of training set (in red color)
plt.scatter(X_train, y_train, color = 'red')

#Create a Linear Regression best fitting line from the training set and predicted values from training set
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
plt.title('X vs y')
plt.show()

#The data is pretty much linear, there's no outliers.
