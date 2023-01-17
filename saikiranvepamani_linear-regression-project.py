import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
customers = pd.read_csv('../input/salaries/Ecommerce Customers')
customers.head()
customers.describe()
customers.info()
sns.jointplot(x='Time on Website' , y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App' , y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App' , y='Length of Membership',kind='hex',data=customers)
sns.pairplot(customers,kind='reg')
sns.lmplot(x="Yearly Amount Spent",y="Length of Membership",data=customers)
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.3, random_state=22)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
print('Coefficients',model.coef_)
prediction  = model.predict(X_test)
model.score(X_train,Y_train)
sns.scatterplot(x=Y_test,y=prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(Y_test,prediction))
print('MSE:',metrics.mean_squared_error(Y_test,prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test,prediction)))
sns.distplot((Y_test-prediction),bins=50)
coeffecients = pd.DataFrame(model.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients