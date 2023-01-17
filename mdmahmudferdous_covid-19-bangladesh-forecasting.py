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
bd=train[(train.Country_Region=='Bangladesh')]

bd
bd_cc=bd[['Date','ConfirmedCases']]
bd_cc['ds']=bd_cc['Date']

bd_cc['y']=bd_cc['ConfirmedCases']

bd_cc.drop(columns=['Date','ConfirmedCases'], inplace=True)

bd_cc.head()
from fbprophet import Prophet

model_cc=Prophet()

model_cc.fit(bd_cc)
future = model_cc.make_future_dataframe(periods=100)

future.head()
forecast=model_cc.predict(future)

forecast.head(5)
fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Confirmed")
bd_ft=bd[['Date','Fatalities']]

bd_ft['ds']=bd_ft['Date']

bd_ft['y']=bd_ft['Fatalities']

bd_ft.drop(columns=['Date','Fatalities'], inplace=True)

bd_ft.head()
from fbprophet import Prophet

model_ft=Prophet()

model_ft.fit(bd_ft)
future = model_ft.make_future_dataframe(periods=100)
forecast=model_ft.predict(future)
fig_Fatalities = model_ft.plot(forecast,xlabel = "Date",ylabel = "Deaths")