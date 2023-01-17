#importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#saving data from csv file into dataframe

customers = pd.read_csv('../input/Ecommerce Customers')
customers.head()
customers.info()
customers.describe()
#Performing exploratory analysis

#analyzing yearly amount spent vs time on website

sns.set_style('darkgrid')

sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent'], data=customers)
#analyzing yearly amount spent vs time on app

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
#analyzing yearly amount spent vs the length of membership

sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
#analyzing these types of relationships all across the data set

sns.pairplot(customers)
#Creating a linear model using seaborn to plot the relationship between Yearly Amount Spent vs Length of Membership

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
#Training and testing data

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training the model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)
#printing coefficients

lm.coef_
predictions = lm.predict(X_test)

sns.scatterplot(y = predictions, x = y_test)

plt.ylabel('Predicted Y')

plt.xlabel('y test')
#Evaluating the performance of the model by calculating the residual sum of squares and the variance score R^2

from sklearn import metrics

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions), ', Mean Squared Error: ',

      metrics.mean_squared_error(y_test, predictions) , ', Root Mean Squared Error: ', 

      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('R^2: Variance Score is ', metrics.explained_variance_score(y_test, predictions))

#This means the model explains nearly 99% of the variance.
#Plotting a histogram of residuals

sns.distplot(y_test-predictions)
#Analyzing coefficients

pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])