import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt1

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import datetime as dt



data = pd.read_csv("/kaggle/input/reliancens/RELIANCE.csv")

data = data.fillna(0) 

data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].map(dt.datetime.toordinal)

tbl = pd.DataFrame(data,columns=['Date','Open','High','Low','Last'])

pd.set_option('display.max_rows', tbl.shape[0]+1)

tbl = tbl.sort_values(['Date'],ascending=False)



#X = tbl.iloc[:, :-1].values

#y = tbl.iloc[:, 1].values

X = tbl['Low'].values.reshape(-1,1)

y = tbl['High'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#print(X_train)



regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)



# Visualising the Training set results

plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, regressor.predict(y_pred), color='red', linewidth=2)

plt.title('Reliance Stocks Prediction')

plt.xlabel('Low')

plt.ylabel('High')

plt.show()



#bar chart

dfPred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#print(dfPred)

dfPred.plot(kind='bar',figsize=(10,8))

plt1.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt1.show()



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))