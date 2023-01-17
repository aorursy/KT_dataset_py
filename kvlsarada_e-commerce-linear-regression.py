# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cust_df = pd.read_csv('/kaggle/input/ecommerce-customers/Ecommerce Customers.csv')

cust_df.head()
pd.unique(cust_df.Avatar)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="Avg. Session Length", y="Yearly Amount Spent", data=cust_df, ax=ax)
sns.scatterplot(x="Time on App", y="Yearly Amount Spent", data=cust_df)
sns.scatterplot(x="Time on Website", y="Yearly Amount Spent", data=cust_df)
sns.scatterplot(x="Length of Membership", y="Yearly Amount Spent", data=cust_df)
# Lets plot a histogram to understand the distribution of data in these features

cust_df.hist(figsize = (12,15))

plt.show()
# Lets create a pair plot 

sns.pairplot(cust_df)

plt.show()
Y = cust_df['Yearly Amount Spent'].values
X = cust_df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
#Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 123)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#coefficients of the model

print('Coefficients: \n',regressor.coef_)
y_pred = regressor.predict(X_test)
plt.scatter(y_test,y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn import metrics

print('MAE= ', metrics.mean_absolute_error(y_test,y_pred))

print('MSE= ', metrics.mean_squared_error(y_test,y_pred))

print('RMSE= ', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

from sklearn import metrics
all_accuracies = cross_val_score(estimator=regressor, X= X, y= Y, cv=5)
print(all_accuracies)
predictions = cross_val_predict(regressor, X, Y, cv=5)

plt.scatter(Y, predictions)
from sklearn import metrics

print('MAE= ', metrics.mean_absolute_error(Y,predictions))

print('MSE= ', metrics.mean_squared_error(Y,predictions))

print('RMSE= ', np.sqrt(metrics.mean_squared_error(Y,predictions)))
all_accuracies2 = cross_val_score(estimator=regressor, X= X, y= Y, cv=8)
print(all_accuracies2)
predictions2 = cross_val_predict(regressor, X, Y, cv=8)

plt.scatter(Y, predictions2)
from sklearn import metrics

print('MAE= ', metrics.mean_absolute_error(Y,predictions2))

print('MSE= ', metrics.mean_squared_error(Y,predictions2))

print('RMSE= ', np.sqrt(metrics.mean_squared_error(Y,predictions2)))