import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import seaborn as sn
data = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')
x = data[['Temperature','Pressure','Humidity','WindDirection(Degrees)','Speed']]

y = data['Radiation']
x['Temperature2'] = x['Temperature']**2

x['Pressure2'] = x['Pressure']**2
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.20,shuffle=True, random_state=236)
model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)
plt.plot(y_test.values[:200])

plt.plot(predictions[:200])
mean_squared_error(y_test, predictions)
model.predict([[57, 30.43, 77,123.6,7.87,3249,925.98]])
x_test.iloc[1]
y_test.iloc[1]