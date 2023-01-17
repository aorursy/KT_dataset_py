# This is simple Linear Regression model application as dataset is simple. It has one independent 

# variable(Crude price) and response variable(dependent variable) is the price/gallon.

# The current prices of the Crude oil and Price/gal can be checked out at below URL:



# https://ycharts.com/indicators/new_york_harbor_ultralow_sulfur_no_2_diesel_spot_price

# https://ycharts.com/indicators/conventional_gasoline_spot_price

    

    

# After a month,I have re-rum my model on latest data. This model predicting the pretty accurate Price/gal.







# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!pip install chart-studio 1.0.0

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


import seaborn as sns

import plotly.graph_objs as go

import plotly

import cufflinks

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
fuelprices = pd.read_csv('../input/transportation-fuels-spot-prices-beginning-2006.csv')
fuelprices.head(10)
fuelprices.isnull().sum()
#drop NY Jetfuel

fuelprices.drop(['NY Jet Fuel Price ($/gal)'],axis=1,inplace=True)
fuelprices.columns

fuelprices.info()
fuelprices.describe()
#When data is symmetrical ,we can apply the Linear Regression algorithm

sns.pairplot(fuelprices)
sns.jointplot(x='WTI Crude Oil Spot Price ($/barrel)',y='NY Conventional Gasoline Spot Price ($/gal)',data=fuelprices)

sns.lmplot(x='WTI Crude Oil Spot Price ($/barrel)',y='NY Conventional Gasoline Spot Price ($/gal)',data=fuelprices)
sns.jointplot(x='Brent Crude Oil Spot Price ($/barrel)',y='NY Ultra-Low Sulfur Diesel Spot Price ($/gal)',data=fuelprices)

sns.lmplot(x='Brent Crude Oil Spot Price ($/barrel)',y='NY Ultra-Low Sulfur Diesel Spot Price ($/gal)',data=fuelprices)
NYConventionalGasoline  = go.Scatter( x=fuelprices['Date'], y=fuelprices['NY Conventional Gasoline Spot Price ($/gal)'],name = 'NYConventionalGasoline')

NYUltraLowSulfurDiesel = go.Scatter( x=fuelprices['Date'], y=fuelprices['NY Ultra-Low Sulfur Diesel Spot Price ($/gal)'],name='NYUltraLowSulfurDiesel')

data = [NYConventionalGasoline, NYUltraLowSulfurDiesel]

layout = dict(title = 'NY Conventional Gasoline vs NY Ultra-Low Sulfur Diesel Price',

              xaxis= dict(title= 'Years',ticklen= 4,zeroline= False))

fig =dict(data = data, layout = layout)

plotly.offline.iplot(fig)
x=fuelprices['NY Conventional Gasoline Spot Price ($/gal)']

sns.distplot(x,bins=50)
WTICrudeOil= go.Bar(x=fuelprices['Date'],y=fuelprices['WTI Crude Oil Spot Price ($/barrel)'],name='WTI Crude Oil',marker= dict(color='red'))

BrentCrudeOil = go.Bar(x=fuelprices['Date'],y=fuelprices['Brent Crude Oil Spot Price ($/barrel)'],name='Brent Crude Oil',marker= dict(color='green'))

data=[WTICrudeOil,BrentCrudeOil]

layout = go.Layout(barmode = "group", title='WTI Crude Oil Price/Barrel VS Brent Crude Oil Price/Barrel ',)

fig = go.Figure(data = data, layout = layout)

plotly.offline.iplot(fig)
X=fuelprices[['WTI Crude Oil Spot Price ($/barrel)']]

y=fuelprices['NY Conventional Gasoline Spot Price ($/gal)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train.head()
X_test.head()
#30% of Total dataset( 638 )

X_test.shape
#70% is the traning data

print(X_train.shape)
lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)


plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()


print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=50)
print(lm.coef_)

print(lm.intercept_)
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

print(cdf)
# Traiing and testing data

X1=fuelprices[['Brent Crude Oil Spot Price ($/barrel)']]

y1=fuelprices['NY Ultra-Low Sulfur Diesel Spot Price ($/gal)']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=101)
X_train1.head()
X_test1.head()

X_train1.shape
#30% of total dataset(638)

X_test1.shape
lm1 = LinearRegression()

lm1.fit(X_train1,y_train1) 
predictions1 = lm.predict(X_test1)


plt.scatter(y_test1,predictions1)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()


print('MAE:', metrics.mean_absolute_error(y_test1, predictions1))

print('MSE:', metrics.mean_squared_error(y_test1, predictions1))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test1, predictions1)))

sns.distplot((y_test1-predictions1),bins=50)
print(lm1.coef_)

print(lm.intercept_)

cdf1=pd.DataFrame(lm1.coef_,X1.columns,columns=['Coeff'])

print(cdf1)