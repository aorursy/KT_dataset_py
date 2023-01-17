import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

import scipy.stats as stats

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
ECommCust = pd.read_csv('../input/Ecommerce Customers')
ECommCust.head()
ECommCust.info()
ECommCust.describe()
sns.set(style="whitegrid", palette='GnBu_d')
j = sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=ECommCust, kind='reg', scatter_kws={"s": 10})

j.annotate(stats.pearsonr)

plt.show()
j = sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=ECommCust, kind='reg', scatter_kws={"s": 10})

j.annotate(stats.pearsonr)

plt.show()
j = sns.jointplot(x='Time on App', y='Length of Membership', data=ECommCust, kind='hex')

j.annotate(stats.pearsonr)

plt.show()
sns.pairplot(ECommCust)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=ECommCust, scatter_kws={"s": 10})
X = ECommCust[['Avg. Session Length', 

               'Time on App',

               'Time on Website', 

               'Length of Membership']]

y = ECommCust['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficients: \n', lm.coef_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions), bins=50)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

coeff_df