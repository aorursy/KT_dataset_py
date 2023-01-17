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
import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns



global_confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

global_deaths = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')



country_names = global_confirmed['Country/Region'].unique().tolist()

country_names
corr = global_confirmed.corr(method='kendall')

plt.figure(figsize=(12,12))

sns.heatmap(corr, annot=True);
countries= ['India', 'US', 'Italy','United Kingdom','Canada', 'Brazil']

i = global_confirmed.loc[global_confirmed['Country/Region']=='India'].iloc[0,4:]  

data1 = pd.DataFrame({'India':i})   

for c in countries:    

    data1[c] = global_confirmed.loc[global_confirmed['Country/Region']==c].iloc[0,4:]

    plt.plot(range(i.shape[0]),data1[c],label=c)

plt.title('Total Number of confirmed cases')

plt.xlabel('Day')

plt.ylabel('Number of Cases')

plt.legend(loc="best")

plt.show()
def confirmed_India_cases():

  confirmed_cases_India = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

  df = pd.read_csv(confirmed_cases_India)

  df = df[df['Country/Region'] == 'India']         

  df_new = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'])

  df_new.rename(columns={"variable":"Date","value":"confirmed_cases"},inplace=True)

  confirmed_per_day = df_new.groupby("Date")['confirmed_cases'].sum()

  confirmed_per_day = confirmed_per_day.reset_index()

  print(confirmed_per_day)    

  confirmed_per_day = confirmed_per_day[['Date','confirmed_cases']]

  return confirmed_per_day
confirmed_cases = confirmed_India_cases()
confirmed_cases.rename(columns={"Date":"ds","confirmed_cases":"y"},inplace=True)

confirmed_cases['ds'] = pd.to_datetime(confirmed_cases['ds'])

confirmed_cases.sort_values(by='ds',inplace=True)



confirmed_plot = confirmed_cases.reset_index()['y'].plot(title='Number of Confirmed Cases in India');

confirmed_plot.set(xlabel='Date', ylabel='Confirmed Cases');
## Split the data in train and test set



X_train = confirmed_cases[:-4]

X_test = confirmed_cases[-4:]



X_test = X_test.set_index("ds")

X_test = X_test['y']
from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py



pred = Prophet()

pred.fit(X_train)

future_dates = pred.make_future_dataframe(periods=7)



forecast =  pred.predict(future_dates)

fig = plot_plotly(pred, forecast)

py.iplot(fig)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plot1 = pred.plot_components(forecast)
from fbprophet.diagnostics import cross_validation



df_cv = cross_validation(pred, initial='15 days', period='15 days', horizon = '60 days')

df_cv.head()
df_cv.tail()
from fbprophet.plot import plot_cross_validation_metric



fig = plot_cross_validation_metric(df_cv, metric='rmse')