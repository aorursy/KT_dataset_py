import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

sns.set_style('whitegrid')
customers = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
customers.head()
customers.info()
customers.shape
customers.describe().transpose()
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers,color='grey')
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers,color='grey')
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,color='grey',kind='hex')
sns.pairplot(data=customers)
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
X = customers[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients:',lm.coef_)

print('\n')

print('Intercept:', lm.intercept_)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics


print('MAE:',metrics.mean_absolute_error(y_test,predictions))

print('MSE:',metrics.mean_squared_error(y_test,predictions))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
sns.distplot((y_test-predictions),bins=45,color='grey')
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients