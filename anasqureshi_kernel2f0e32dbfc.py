import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



customers = pd.read_csv('../input/Ecommerce Customers.csv')

customers.head()
sns.jointplot(data=customers,x='Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')
#training and testing data

customers.columns
X = customers.drop(['Yearly Amount Spent','Email','Address','Avatar'],axis=1)

X.head()
y = customers['Yearly Amount Spent']

y.head()
!pip install sklearn

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)



X_train



#now we will train the model
from sklearn.linear_model import LinearRegression 

model = LinearRegression()

model.fit(X_train,y_train)
model.coef_
model.intercept_
predictions = model.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y test (true values)')

plt.ylabel('predicted values')
from sklearn import metrics

print('MAE', metrics.mean_squared_error(y_test,predictions))

print('MSE', metrics.mean_absolute_error(y_test,predictions))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
metrics.explained_variance_score(y_test,predictions)
#residual

sns.distplot(y_test-predictions,bins = 50)
#conclusion : do we focus our efforts on mobile app or website development ?



pd.DataFrame(model.coef_,X.columns,columns=['coeff'])



#answers : 1) company should work on website to catch up with the app

          # 2) company can improve the app more as 1 unit increase in app leads to 38$ increase 

    # 3) overall it now all depends on the product owner 