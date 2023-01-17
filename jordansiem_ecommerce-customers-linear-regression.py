import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
%matplotlib inline
customers = pd.read_csv("../input/Ecommerce Customers.csv")
customers.head()
customers.describe()
customers.info()
sns.jointplot(data=customers,x='Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
sns.pairplot(customers)
#Length of Membership
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
customers.columns
y = customers['Yearly Amount Spent']
# Y we are trying to predict
x = customers[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]
#Split to training and testing set.
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
#Train Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
lm.coef_
#Predicting
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test (True Values)')

plt.ylabel('Predicted Values')
#Evaluate Model
#Calculating model performance and residuals
from sklearn import metrics
print('MAE ',metrics.mean_absolute_error(y_test,predictions))

print('MSE ',metrics.mean_squared_error(y_test,predictions))

print('RMSE ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
metrics.explained_variance_score(y_test,predictions)
#How much variance model is explaining. 98% - Fit model
#Residuals
sns.distplot((y_test-predictions),bins = 50)
cdf = pd.DataFrame(lm.coef_,x.columns,columns = ['Coeff'])

cdf
#How to interpret? One at time look at. 
#Hold all fixed. 1 unit increase = Avg Session Length of 26 dollars more spent

#Increase of $38. Length of membership is highest. Website needs most work?

#Develop App more? Doing better. Explore relationship of membership and app.

#Hard to tell without knowing the costs.