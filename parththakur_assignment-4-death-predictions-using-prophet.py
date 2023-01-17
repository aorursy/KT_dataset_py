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
deaths = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
deaths_canada = deaths.loc[deaths['Country/Region']=='Canada'].iloc[:,4:]
deaths_canada


sum_df = deaths_canada.sum()

deaths_canada.loc['Total']= sum_df

deaths_canada
from matplotlib import pyplot as plt

sum_df.plot(kind = 'bar', figsize = (50,20))

col_names = list(deaths_canada.columns.values)

deaths = list(deaths_canada.T['Total'])

prophet_df = pd.DataFrame(columns = ['ds','y'])

prophet_df['ds']=col_names

prophet_df['y']= deaths



prophet_df
from fbprophet import Prophet



m = Prophet(interval_width=0.95)

m.fit(prophet_df)

future = m.make_future_dataframe(periods=15)



forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
m.plot(forecast)