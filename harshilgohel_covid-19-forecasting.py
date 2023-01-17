import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import datetime

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
train.head()
train_data_by_country = train.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum'})

max_train_date = train['Date'].max()

train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)

train_data_by_country_confirm.set_index('Country_Region', inplace=True)

train_data_by_country_confirm.style.background_gradient(cmap='Blues').format({'ConfirmedCases': "{:.0f}"})
confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

confirmed_total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global Fatalities cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
train['Province_State'].fillna("none",inplace=True)

train['ConfirmedCases'] = train['ConfirmedCases'].astype(int) 

train['Fatalities'] = train['Fatalities'].astype(int)
train['Date'] = pd.to_datetime(train['Date'])

train['Day']=train['Date'].dt.day

train['Month']=train['Date'].dt.month
lbl=LabelEncoder()

train['Province_State']=lbl.fit_transform(train['Province_State'])

train['Country_Region']=lbl.fit_transform(train['Country_Region'])
X = train[['Province_State','Country_Region','Day','Month']]

X.head()
y = train[['ConfirmedCases','Fatalities']]

y.head()
tree=DecisionTreeRegressor()

tree.fit(X,y)
test.head()
test['Province_State'].fillna("none",inplace=True)



test['Date'] = pd.to_datetime(test['Date'])

test['Day']=test['Date'].dt.day

test['Month']=test['Date'].dt.month



test['Province_State']=lbl.fit_transform(test['Province_State'])

test['Country_Region']=lbl.fit_transform(test['Country_Region'])
X_test = test[['Province_State','Country_Region','Day','Month']]

X_test.head()
prediction=tree.predict(X_test)

result = pd.DataFrame(prediction)

result.columns = ['ConfirmedCases','Fatalities']

result.head()