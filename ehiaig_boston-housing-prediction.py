#Import the necessary packages
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
#Load the dataset
data = pd.read_csv('../input/boston_train.csv')
#Show the first 5 rows in the csv
data.head()
#Get an overview of the data to see if there are any empty values in the dataset
data.info()
#Let's to get the dimension of the dataset.
data.shape
#Now let's get a summary of our data to see the distribution.
data.describe()
# Get the titles of the columns in our dataset
data.columns
# The columns we'll use to predict the target
X = data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio',
 'black', 'lstat']]
#OR X = data.drop('medv', axis=1)

# The target column
Y = data['medv']
# Split our prepared data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)
#Let's fit our model using Linear Regression
from sklearn.linear_model import LinearRegression

#Create an instance of LinearRegression
lm = LinearRegression()
#fit the model on our training data
lm.fit(X_train, Y_train)

#let's grab the predictions from our test set
Y_predictions = lm.predict(X_test)
# Plot the Actual Prices against the Predicted Prices
plt.scatter(Y_test, Y_predictions)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title("Plot of Actual prices VS Predicted prices")
#Checking the residual
sns.distplot((Y_test - Y_predictions))
# MODEL EVALUATION
from sklearn import metrics

MAE = metrics.mean_absolute_error(Y_test, Y_predictions)

MSE = metrics.mean_squared_error(Y_test, Y_predictions)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_predictions))

print('MAE: ', MAE )
print('MSE: ', MSE)

print('RMSE: ', RMSE)

