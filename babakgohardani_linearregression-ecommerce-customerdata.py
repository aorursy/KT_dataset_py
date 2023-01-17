import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

# Any results you write to the current directory are saved as output.
customers = pd.read_csv('../input/Ecommerce Customers')
customers.head(5)
customers.describe()
sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')
sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')
sns.jointplot(data=customers, x='Time on App', y='Length of Membership')
sns.pairplot(customers)
plt.scatter(customers['Length of Membership'], customers['Yearly Amount Spent'])

plt.xlabel('Length of Membership')

plt.ylabel('Yearly Amount Spent')
customers.columns
Y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
X.columns.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
lm = LinearRegression()

lm.fit(X_train, Y_train)
lm.coef_
preds = lm.predict(X_test)
plt.scatter(Y_test, preds)

plt.xlabel('Y true values')

plt.ylabel('Y predictions')
print('MAE', metrics.mean_absolute_error(Y_test, preds))
sns.distplot((Y_test-preds), bins=30)
pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])