import os

import pandas as pd

from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
data = pd.read_csv('../input/weatherww2/Summary of Weather.csv')
data.shape
data.head()
data.describe()
data.plot(x='MinTemp', y='MaxTemp', style='o')  

plt.title('MinTemp vs MaxTemp')  

plt.xlabel('MinTemp')  

plt.ylabel('MaxTemp')  

plt.show()
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(data['MaxTemp'])
df = data.loc[:,['MinTemp','MaxTemp']]

df.shape
x = df['MinTemp'].values.reshape(-1,1)

y = df['MaxTemp'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)
regressor = LinearRegression()

regressor.fit(x_train, y_train) #training the algorithm
#To retrieve intercept

print(regressor.intercept_)



#to retrieve slope

print(regressor.coef_)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()})

df.shape

df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#Plot straight line with the test data

plt.scatter(x_test,y_test,color='grey')

plt.plot(x_test,y_pred,color='red',linewidth=2)
#Find the value of these metrices

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test,y_pred)
plt.scatter(x_test,y_test, color='red')

plt.plot(x_test,regressor.predict(x_test))

plt.xlabel('Minimum Temperature')

plt.ylabel('Maximum Temperature')

plt.title('Minimum Vs Maximum Temperature')

plt.show()
np.savetxt('Univariate_predicted.csv',df,delimiter=',')