import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data1 = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers.csv')
data1.head()
data1.describe()
data1.info()
sns.set_palette('RdYlGn',10)

sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data =data1)
sns.jointplot(x='Time on App',y ='Yearly Amount Spent' ,data= data1)
sns.jointplot(x='Time on App',y ='Length of Membership', kind='hex',data = data1)
sns.pairplot(data1)
sns.lmplot(x='Length of Membership',y ='Yearly Amount Spent', data = data1)
X= data1.iloc[:,[3,4,5,6]]

Y= data1.iloc[:,-1]
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3, random_state=100)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)
lr.coef_
predictions = lr.predict(X_test)
plt.scatter(Y_test, predictions , color='red')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(Y_test, predictions))

print('MSE :'," ",metrics.mean_squared_error(Y_test, predictions))

print('RMAE:'," ", np.sqrt(metrics.mean_squared_error(Y_test , predictions)))
sns.distplot(Y_test- predictions, bins=40 , color = 'red')