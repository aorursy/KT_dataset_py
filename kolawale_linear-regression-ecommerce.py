import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
customers = pd.read_csv('../input/linear-regression/Ecommerce Customers.csv')
customers.head()
customers.describe()
customers.info()
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y ='Yearly Amount Spent', data = customers)
sns.jointplot(x='Time on App', y='Length of Membership', data = customers, kind = 'hex')
sns.pairplot(customers)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
customers.columns
X= customers[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]



y= customers ['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print (lm.coef_)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics

print (metrics.mean_absolute_error(y_test, predictions))

print (metrics.mean_squared_error(y_test, predictions))

print (np.sqrt (metrics.mean_squared_error(y_test, predictions)))
sns.distplot(y_test-predictions)
pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficient'])
# 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.

# 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.

# 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.

# 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.