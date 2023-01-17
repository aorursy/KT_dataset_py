#Using linear regression to analyze customer behaviour 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ecom_customers = pd.read_csv('../input/Ecommerce Customers')

ecom_customers.head()                             
#Dataset info

ecom_customers.info()
ecom_customers.describe()
#Scatter plot of Yearly Amount Spent vs Time on Website. 

#Since we have a circular shape, there is not a significant correlation between these two variables.

sns.jointplot(ecom_customers['Time on Website'],ecom_customers['Yearly Amount Spent'])
#Scatter plot of Yearly Amount Spent vs Time on App. 

#We can see there is a correlation between the amount of time spent on the app and the amount of money spent.

sns.jointplot(ecom_customers['Time on App'],ecom_customers['Yearly Amount Spent'])
sns.jointplot(ecom_customers['Time on App'],ecom_customers['Length of Membership'],kind='hex')
#A grid plot of all the variables. This shows that lenghth of membership is the most effective factor on yearly amount spent.

sns.pairplot(ecom_customers)
sns.heatmap(ecom_customers.corr(),cmap = 'Blues', annot=True)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data = ecom_customers)
#Using Linear Regression to predict the amount of money spent based on the other variable.



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



x=ecom_customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

y=ecom_customers['Yearly Amount Spent']



#Splitting the data into test data and training data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)



#Creating a Linear Regression model

lm = LinearRegression()

lm.fit(x_train,y_train)

coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])

coeff_df
#Predicting the output using the model

predictions = lm.predict(x_test)

plt.scatter(y_test,predictions)
#Evaluating the results usind different metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#plotting the distribution of the prediction erros. ]

#Since the error has a normal distribution, the model is considered suitable for the data.

sns.distplot((y_test-predictions),bins=50);