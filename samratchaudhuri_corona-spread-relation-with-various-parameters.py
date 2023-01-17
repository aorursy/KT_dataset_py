import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv(r'/kaggle/input/corona-spread-dataset/Corona spread data.csv')

df.head()
df.shape
df.info()
sns.boxplot(df['conf'])

plt.show()
sns.pairplot(df, x_vars=['Temp', 'Humid', 'Pop den'], y_vars='conf', height=4, aspect=1, kind='scatter')

plt.show()
sns.pairplot(df, x_vars=['BPL', 'PCS H', 'PC I','Lit rate'], y_vars='conf', height=4, aspect=1, kind='scatter')

plt.show()
sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)

plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



a= df.groupby('Temp').conf.mean().reset_index()

print (a) 



y= a.conf

features=['Temp']

X= a[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



dependecy_model= RandomForestRegressor(random_state=1)

dependecy_model.fit(train_X, train_y)

pred= dependecy_model.predict(val_X)

print(mean_absolute_error(val_y, pred))



regr= LinearRegression()

regr.fit(X,y)

y_pred = regr.predict(X)



plt.plot(X, y_pred)  #The Blue Graph Shows the current trends.

plt.show()



X_future = np.arange(36, 51)    #The Red Graph Shows the future,i.e. upcoming trends.

X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)



plt.plot(X_future, future_predict, 'red')

plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



b = df.groupby('Humid').conf.mean().reset_index()

print (b)



y= b.conf

features=['Humid']

X= b[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



dependecy_model= RandomForestRegressor(random_state=1)

dependecy_model.fit(train_X, train_y)

pred= dependecy_model.predict(val_X)

print(mean_absolute_error(val_y, pred))



regr= LinearRegression()

regr.fit(X,y)

y_pred = regr.predict(X)



plt.plot(X, y_pred)   #The Blue Graph Shows the current trends.

plt.show()



X_future = np.arange(82, 101)    #The Red Graph Shows the future,i.e. upcoming trends.

X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)



plt.plot(X_future, future_predict, 'red')

plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



b = df.groupby('PCS H').conf.mean().reset_index()

print(b)



y= b.conf

features=['PCS H']

X= b[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



dependecy_model= RandomForestRegressor(random_state=1)

dependecy_model.fit(train_X, train_y)

pred= dependecy_model.predict(val_X)

print(mean_absolute_error(val_y, pred))



regr= LinearRegression()

regr.fit(X,y)

y_pred = regr.predict(X)



plt.plot(X, y_pred)   #The Blue Graph Shows the current trends.

plt.show()



X_future = np.arange(6001, 10001)     #The Red Graph Shows the future,i.e. upcoming trends.

X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)



plt.plot(X_future, future_predict, 'red')

plt.show()