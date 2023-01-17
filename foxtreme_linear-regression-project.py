import pandas as pd

import numpy as np

import matplotlib as plt

import seaborn as sns

%matplotlib inline
import os

print(os.listdir("../input/"))
customers = pd.read_csv('../input/Ecommerce Customers')
customers.head()
customers.describe()
customers.info()
sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = customers)
sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = customers)
sns.jointplot(x='Time on App',y ='Length of Membership', data = customers, kind='hex')
sns.pairplot(customers)
print("Length of Membership")
sns.set(color_codes=True)

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)
X = customers[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]
y= customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train )
print(lm.coef_)
predictions = lm.predict(X_test)
plt.pyplot.scatter(y_test, predictions)

plt.pyplot.ylabel('Predicted')

plt.pyplot.xlabel('Y test')
import sklearn.metrics as metrics

print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predictions)))

print('MSE: {}'.format(metrics.mean_squared_error(y_test, predictions)))

print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))
sns.distplot((y_test-predictions))
pd.DataFrame(lm.coef_ , X.columns, columns=['Coeffecient'])