import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Importing our dataset from CSV file:

dataset = pd.read_csv("../input/student-scores/student_scores.csv")



# Now let's explore our dataset:

dataset.shape
# Let's take a look at what our dataset actually looks like:

dataset.head()
# To see statistical details of the dataset, we can use describe():

dataset.describe()
#And finally, let's plot our data points on 2-D graph our dataset 

#and see if we can manually find any relationship between the data:



dataset.plot(x='Hours', y='Scores', style='o')

plt.title('Hours vs Score')

plt.xlabel('Hours Studied')

plt.ylabel('Percentage Score')

plt.show()

# Preparing our data:

# Divide the data into "attributes" and "labels". Attributes are the independent variables

# while labels are dependent variables whose values are to be predicted.



X = dataset.iloc[:, :-1].values # all colomns except the last one (reshape it into column vector)

 

y = dataset.iloc[:, 1].values # first colomn only

''' 

The next step is to split this data into training and test sets. 

We'll do this by using Scikit-Learn's built-in train_test_split() method:

'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# The above script splits 80% of the data to training set while 20% of the data to test set. 

# The test_size variable is where we actually specify the proportion of test set.
# Training the Algorithm:



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# To retrieve the intercept:

print(regressor.intercept_)



# For retrieving the slope (coefficient of x):

print(regressor.coef_)

# Making Predictions:

# Now that we have trained our algorithm, it's time to make some predictions.



y_pred = regressor.predict(X_test)    # The y_pred is a numpy array that contains all the predicted values.

# To compare the actual output values for X_test with the predicted values, execute the following script:



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df

# Plot actual value vs predicted one:



plt.scatter(X_test, y_test)

plt.plot(X_test, y_pred, color='red')



plt.title('Hours vs Percentage')

plt.xlabel('Hours Studied')

plt.ylabel('Percentage Score')

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2_score: ', metrics.r2_score(y_test,y_pred))