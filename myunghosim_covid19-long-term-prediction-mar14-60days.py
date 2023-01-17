import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
import os

print(os.listdir("../input/"))
df = pd.read_csv("../input/covid-19-mar14/covid_19_data.csv")

df
usa = df.loc[df['Country/Region'] == 'US']

usa.max()
usa
usa.min()
confirmed_us = usa.groupby('ObservationDate')['Confirmed'].sum().reset_index()
confirmed_us = confirmed_us.set_index('ObservationDate')

confirmed_us.index
confirmed_us
size = len(confirmed_us)

us_change = [confirmed_us['Confirmed'][i]-confirmed_us['Confirmed'][i-1] for i in range(1,size)]
confirmed_us.plot(figsize=(15, 6))

plt.show()
days=[x+1 for x in range(size-1)]

plt.plot(days, us_change, color='red', linewidth=2)

plt.show()
import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
X = np.asarray([x for x in range(len(us_change))]).reshape(-1,1)

y = np.asarray(us_change).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
size = len(confirmed_us)

X_pred = np.asarray([x for x in range(size,size+60)]).reshape(-1,1)
y_pred = regressor.predict(X_pred)
#plot predictions for change of confirmed cases

plt.scatter(X_train, y_train,  color='gray')

plt.plot(X_pred, y_pred, color='red', linewidth=2)

plt.xlabel('Nth Day of Coronavirus in US')

plt.ylabel('Number of NEW Confirmed Cases(Daily)')

plt.title('Predicted NEW confirmed coronavirus cases in US - rate of change')

plt.show()
y_list=[]

cumul_sum=0

for y in list(y_pred.flatten()):

    cumul_sum+=y

    y_list.append(cumul_sum)
total_y = list(y_train.flatten())

total_y.extend(y_list)

total_x = [x for x in range(len(total_y))]
#plot predictions for cumulative confirmed cases

x_orig = [x for x in range(len(confirmed_us))]

plt.scatter(x_orig, confirmed_us,  color='gray')

plt.plot(total_x, total_y, color='red', linewidth=2)

plt.xlabel('Nth Day of Coronavirus in US')

plt.ylabel('Number of Confirmed Cases(Daily)')

plt.title('Predicted confirmed coronavirus cases in US')

plt.show()