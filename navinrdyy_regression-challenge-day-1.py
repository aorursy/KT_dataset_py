import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from pandas import set_option
bicycle = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
bicycle.head()
bicycle = bicycle.rename(columns={'Unnamed: 0' : 'Record Number'})

bicycle = bicycle.set_index('Date')

bicycle.drop('Day',axis=1,inplace=True)

bicycle.head(3)
bicycle.dtypes
temp = bicycle['Precipitation'].replace(['0.47 (S)'], '0.47')

temp = temp.replace(['T'], '0')

temp = temp.astype(float)
bicycle['Precipitation'] = temp

bicycle.head(4)
bicycle.Precipitation.value_counts()
set_option('precision',1)

bicycle.describe()
set_option('precision',2)

x = bicycle.corr()

x
plt.subplots(figsize=(10,6))

sns.heatmap(x,cmap='RdYlGn',annot = True)

plt.show()
temp = bicycle['High Temp (°F)'].describe()

temp
bicycle['High Temp <25%'] = (bicycle['High Temp (°F)'] <= temp['25%']).astype(int)

bicycle['High Temp >25&<50%'] = ((bicycle['High Temp (°F)'] > temp['25%']) & (bicycle['High Temp (°F)'] <= temp['50%'])).astype(int)

bicycle['High Temp >50&<75%'] = ((bicycle['High Temp (°F)'] > temp['50%']) & (bicycle['High Temp (°F)'] <= temp['75%'])).astype(int)
bicycle.head()
temp = bicycle['Low Temp (°F)'].describe()

temp
bicycle['Low Temp <25%'] = (bicycle['Low Temp (°F)'] <= temp['25%']).astype(int)

bicycle['Low Temp >25&<50%'] = ((bicycle['Low Temp (°F)'] > temp['25%']) & (bicycle['Low Temp (°F)'] <= temp['50%'])).astype(int)

bicycle['Low Temp >50&<75%'] = ((bicycle['Low Temp (°F)'] > temp['50%']) & (bicycle['Low Temp (°F)'] <= temp['75%'])).astype(int)
bicycle.head(2)
temp = bicycle['Precipitation'].describe()

temp
bicycle.Precipitation.value_counts().sort_index()
bicycle['Precp >75%'] = (bicycle['Precipitation'] >= temp['75%']).astype(int)

bicycle['Precp min'] = (bicycle['Precipitation'] == 0).astype(int)

bicycle['Precp 1'] = (bicycle['Precipitation'] < 0.09).astype(int)

bicycle['Precp 2'] = (bicycle['Precipitation'] <= 0.15).astype(int)

bicycle['Precp 3'] = (bicycle['Precipitation'] <= 0.20).astype(int)
#sns.pairplot(bicycle,x_vars=['High Temp (°F)','Low Temp (°F)','Precipitation'],y_vars='Total',kind='reg',size=4)

#plt.show()
X = bicycle.drop(['Total','Record Number','High Temp (°F)','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge'],axis=1)

y = bicycle['Total']
lin = LinearRegression()
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=1)

print(Xtrain.shape)

print(Xtest.shape)

print(ytrain.shape)

print(ytest.shape)
lin.fit(Xtrain,ytrain)

print(lin.intercept_)

lin.coef_
y_pred = lin.predict(Xtest)

np.sqrt(metrics.mean_squared_error(ytest,y_pred))
df = pd.DataFrame({})

df = pd.concat([Xtest,ytest],axis=1)

df['Predicted'] = np.round(y_pred,2)

df['ERROR'] = df['Total'] - df['Predicted']
df.head(2)
df['ERROR'].describe()