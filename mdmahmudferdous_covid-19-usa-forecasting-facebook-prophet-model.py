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
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

train.head()
us=train[(train.Country_Region=='US')]

us=us.groupby(us.Date).sum()

us['Date']=us.index
us_cc=us[['Date','ConfirmedCases']]

us_cc['ds']=us_cc['Date']

us_cc['y']=us_cc['ConfirmedCases']

us_cc.drop(columns=['Date','ConfirmedCases'], inplace=True)

us_cc.head()
from fbprophet import Prophet

model_cc=Prophet()

model_cc.fit(us_cc)
future = model_cc.make_future_dataframe(periods=100)

future.head()
forecast=model_cc.predict(future)

forecast.tail(5)
fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Confirmed")
us_ft=us[['Date','Fatalities']]

us_ft['ds']=us_ft['Date']

us_ft['y']=us_ft['Fatalities']

us_ft.drop(columns=['Date','Fatalities'], inplace=True)

us_ft.head()
from fbprophet import Prophet

model_ft=Prophet()

model_ft.fit(us_ft)
future = model_ft.make_future_dataframe(periods=100)

forecast=model_ft.predict(future)
fig_Fatalities = model_ft.plot(forecast,xlabel = "Date",ylabel = "Deaths")