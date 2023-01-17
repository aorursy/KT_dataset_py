import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/e-commerce-customer/Ecommerce Customers.csv')
data.head(2)
data.isnull().sum()
data.describe()
data.info()
sns.set()
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=data)

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=data)

sns.jointplot(x='Time on App', y='Length of Membership', data=data, kind='hex')

sns.pairplot(data)

sns.regplot(x='Length of Membership', y='Yearly Amount Spent', data=data)
data.corr()
data.columns
x = data[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = data['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y,  test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
a = lm.fit(x_train,y_train)
a.score(x_train, y_train)
lm.coef_
x.columns
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(x_test)
predictions[0]
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

model.score(x, y)
sns.distplot((y_test-predictions),bins=50);
coeff_df = pd.DataFrame(model.coef_,x.columns,columns=['Coefficient'])
coeff_df
