# Import libraries

import pandas as pd

import numpy as np
# Import visualisation libraries



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
# Read the salary dataset



salary = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
# Check the head of the data

salary.head(3)
# Check the info of the data

salary.info()
# Check the shape of the data

salary.shape
# Check the description of the data

salary.describe()
# Check for nulls 

salary.isnull().sum()
# Use seaborn jointplot to see the relationships

sns.jointplot(x='YearsExperience',y='Salary',data=salary)
sns.heatmap(salary.corr(), annot=True, cmap='viridis')
sns.lmplot(x='YearsExperience',y='Salary',data=salary)
# Training and Testing Data

X = np.array(salary.YearsExperience).reshape(-1,1)

y = np.array(salary.Salary).reshape(-1,1)
# Use model_selection.train_test_split from sklearn to split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Training the model

from sklearn.linear_model import LinearRegression
# Create an instance of a LinearRegression model

lm = LinearRegression()

lm.fit(X_train,y_train)
# Print out the coefficients and intercept of the model

print('Coefficient:\n', lm.coef_)

print('Intercept:\n',lm.intercept_)
# Predicting Test Data

predictions = lm.predict(X_test)
# Scatter plot of the test values and the predicted values

plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# Evaluating the model

from sklearn import metrics



print('MAE:',metrics.mean_absolute_error(y_test,predictions))

print('MSE:',metrics.mean_squared_error(y_test,predictions))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
# Residuals

sns.distplot((y_test-predictions),bins=50)