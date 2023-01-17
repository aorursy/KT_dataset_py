import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
customers = pd.read_csv('../input/Ecommerce Customers')
customers.describe()

customers.info()

customers.head()
sns.jointplot(x=customers['Time on Website'],y=customers['Yearly Amount Spent'])
sns.jointplot(x=customers['Time on App'],y=customers['Yearly Amount Spent'])
sns.jointplot(x=customers['Time on App'],y=customers['Length of Membership'],kind='hex')
sns.pairplot(customers)
#Length of Membership
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
customers.columns

x = customers[[ 'Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
lm.coef_
prediction = lm.predict(x_test)
plt.scatter(prediction,y_test)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,prediction))

print('MSE:', metrics.mean_squared_error(y_test,prediction))

print('MSE:', np.sqrt(metrics.mean_squared_error(y_test,prediction)))



sns.distplot(prediction-y_test,bins = 50)
df = pd.DataFrame(lm.coef_,index = x.columns, columns = ['Coeffecient'])

df