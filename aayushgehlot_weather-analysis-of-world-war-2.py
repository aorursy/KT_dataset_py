import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as sns

import os

import pandas_profiling as pp

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn import metrics

%matplotlib inline
os.getcwd()
data = pd.read_csv("../input/weatherww2/Summary of Weather.csv", encoding='latin', low_memory=False) 

data.tail(2)
data.shape
pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', None)  # or 1000

pd.set_option('display.max_colwidth', -1)  # or 199
data.describe()
data.isnull().sum()
data=data[['STA','Date','Precip','MaxTemp','MinTemp','MeanTemp','Snowfall',

           'PoorWeather','YR','MO','DA','SNF','TSHDSBRSGF']]
data.info()
pp.ProfileReport(data)
plt.scatter(data.MinTemp, data.MaxTemp,  color='gray')
f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax, cmap="BuPu")

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(data['MaxTemp'], color='orange')

plt.xlabel('Max Temperature')

plt.xlim(0,60)

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(data['MinTemp'], color='Red')

plt.xlabel('Min Temperature')

plt.xlim(0,40)

plt.show()
X = data['MinTemp'].values.reshape(-1,1)

y = data['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(X_test)

y_pred
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df.head(10)
df1 = df.head(20)

df1.plot(kind='bar',figsize=(8,12))

plt.grid(which='both', linestyle='-', linewidth='0.5')

plt.show()
plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
r2_score(y_test, y_pred,multioutput='variance_weighted') 