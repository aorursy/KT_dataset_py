import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
customers = pd.read_csv("../input/new-york-ecommerce-customers/Ecommerce Customers")
customers.head()
customers.describe()
sns.distplot(customers["Yearly Amount Spent"],bins=25)
sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent',data = customers)
customers.corr()
sns.heatmap(customers.corr(),cmap='coolwarm', annot=True,fmt=".2f",annot_kws={'size':16},cbar=False)
sns.pairplot(customers)
sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent',data = customers)
X = customers[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

Y = customers["Yearly Amount Spent"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(x = predictions, y = Y_test)
sns.distplot(Y_test-predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))

print('MSE:', metrics.mean_squared_error(Y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))