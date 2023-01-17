import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
customers = pd.read_csv('../input/Ecommerce Customers')
customers.info()
sns.jointplot(data=customers,x = 'Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customers,x = 'Time on App',y='Yearly Amount Spent')
sns.jointplot(data=customers,x = 'Time on App',y='Length of Membership',kind='hex')
sns.pairplot(customers)
sns.lmplot(data=customers,x='Length of Membership',y='Yearly Amount Spent')
customers.columns
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predictions = lm.predict(X_test)
plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test(True Values)')
plt.ylabel('Predicted Values')
from sklearn import metrics
print('MAE :',metrics.mean_absolute_error(y_test,predictions))
print('MSE :',metrics.mean_squared_error(y_test,predictions))
print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('Variance Score :',metrics.explained_variance_score(y_test,predictions))
plt.figure(figsize=(12,6))
sns.distplot(y_test-predictions,bins=50)
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf
