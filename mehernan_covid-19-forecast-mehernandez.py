# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data into Pandas dataframes

df_train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

df_submission = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
# Check a preview of the data

df_train.tail()
# Check the properties of the data



print(df_train['Province/State'].unique())

print(df_train['Country/Region'].unique())

print(df_train['Lat'].unique())

print(df_train['Long'].unique())

print(df_train.dtypes)
df_train.describe()
# Check the distribution of the confirmed cases



df_train.hist(column='ConfirmedCases')
# Check the distribution of the fatalities



df_train.hist(column='Fatalities')
# Take only what we need: date, confirmed cases and fatalities



df_train = df_train[['Date', 'ConfirmedCases', 'Fatalities']]
# Convert Date column to Pandas date and orther to get chronological data



df_train['Date'] = pd.to_datetime(df_train['Date'])

df_train = df_train.sort_values(by=['Date'])
# Check the trend in a chart



df_train.plot.bar(x='Date', y=['ConfirmedCases','Fatalities'])
# As the confirmed cases are far away from the start we will focus in that time



df_train2 = df_train.query('ConfirmedCases != 0.0')



df_train2.plot.bar(x='Date', y=['ConfirmedCases', 'Fatalities'])
df_train['Week'] = df_train['Date'].dt.week

df_train['Day'] = df_train['Date'].dt.day

df_train['WeekDay'] = df_train['Date'].dt.dayofweek

df_train['YearDay'] = df_train['Date'].dt.dayofyear
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score



from sklearn.model_selection import train_test_split



predictors = df_train.drop(['Date', 'ConfirmedCases', 'Fatalities'], axis=1)

target = df_train[['ConfirmedCases', 'Fatalities']]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=1)



def scores(alg):

    lin = alg()

    lin.fit(x_train, y_train['ConfirmedCases'])

    y_pred = lin.predict(x_test)

    lin_r = r2_score(y_test['ConfirmedCases'], y_pred)

    s.append(lin_r)

    

    lin.fit(x_train, y_train['Fatalities'])

    y_pred = lin.predict(x_test)

    lin_r = r2_score(y_test['Fatalities'], y_pred)

    s2.append(lin_r)

    

algos = [KNeighborsRegressor, LinearRegression, RandomForestRegressor, GradientBoostingRegressor, Lasso, ElasticNet, DecisionTreeRegressor]



s = []

s2 = []



for algo in algos:

    scores(algo)

    

models = pd.DataFrame({

    'Method': ['KNeighborsRegressor', 'LinearRegression', 'RandomForestRegressor', 'GradientBoostingRegressor', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor'],

    'ScoreCC': s,

    'ScoreF' : s2

})



models.sort_values(by=['ScoreCC', 'ScoreF'], ascending=False)
# Now let's try for last an ARIMA model



# First we see that data is not stationary, so we need to check the autocorrelation of the time series



from pandas.plotting import autocorrelation_plot



autocorrelation_plot(df_train['ConfirmedCases'])
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import*



arima_model = ARIMA(df_train['ConfirmedCases'], order=(4,1,0)).fit(disp=0, transparams=True, trend='c')

print(arima_model.summary())



residuals = pd.DataFrame(arima_model.resid)

residuals.plot()

pyplot.show()

residuals.plot(kind='kde')

pyplot.show()

print(residuals.describe())



arima_model2 = ARIMA(df_train['Fatalities'], order=(4,1,0)).fit(disp=0, transparams=True, trend='c')

print(arima_model2.summary())



residuals2 = pd.DataFrame(arima_model2.resid)

residuals2.plot()

pyplot.show()

residuals2.plot(kind='kde')

pyplot.show()

print(residuals2.describe())
predictions_arima = list(arima_model.predict())

predictions_arima.append(arima_model.forecast()[0][0])

predictions_arima.append(arima_model.forecast()[0][0])



df_train['arima'] = predictions_arima



predictions_arima2 = list(arima_model2.predict())

predictions_arima2.append(arima_model2.forecast()[0][0])

predictions_arima2.append(arima_model2.forecast()[0][0])



df_train['arima2'] = predictions_arima2



df_train.plot.bar(x='Date', y=['ConfirmedCases', 'arima'])

df_train.plot.bar(x='Date', y=['Fatalities', 'arima2'])
df_submission.head()
print(df_test['Date'].values)

print(len(df_test['Date']))
df_test = df_test[['ForecastId', 'Date']]



df_test['Date'] = pd.to_datetime(df_test['Date'])

df_test['Week'] = df_test['Date'].dt.week

df_test['Day'] = df_test['Date'].dt.day

df_test['WeekDay'] = df_test['Date'].dt.dayofweek

df_test['YearDay'] = df_test['Date'].dt.dayofyear



df_test.head()
model = RandomForestRegressor()

model.fit(x_train, y_train['ConfirmedCases'])



model2 = RandomForestRegressor()

model2.fit(x_train, y_train['Fatalities'])





df_test['ConfirmedCases'] = model.predict(df_test.drop(['Date', 'ForecastId'], axis=1))

df_test['Fatalities'] = model2.predict(df_test.drop(['Date', 'ForecastId', 'ConfirmedCases'], axis=1))
df_final = df_test[['ForecastId', 'ConfirmedCases', 'Fatalities']] 

df_final['ConfirmedCases'] = df_final['ConfirmedCases'].astype(int)

df_final['Fatalities'] = df_final['Fatalities'].astype(int)



df_final.head()
df_final.plot.bar(x='ForecastId', y=['ConfirmedCases', 'Fatalities'])
df_final.to_csv('submission.csv', index=False)