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

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
confirmed_acc = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/confirmed_acc.csv')

confirmed_acc.head()
cases = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/cases.csv')

cases.head()
patient = pd.read_csv('/kaggle/input/indonesia-coronavirus-cases/patient.csv')

patient.head()
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of Confirmed Person/Patient")

sns.kdeplot(data=patient['age'].dropna(), shade=True)
sns.set_style("darkgrid")

plt.title("Presentation of Confirmed Person by Gender")

patient.groupby('gender')['patient_id'].count().plot.pie(figsize=(10, 20), autopct='%1.1f%%', startangle=90)
sns.set_style("darkgrid")

plt.title("Presentation of Person who have a contacted with The Positive")

patient.groupby('contacted_with')['patient_id'].count().plot.pie(figsize=(10, 20), autopct='%1.1f%%', startangle=90)
patient.groupby(['contacted_with', 'gender'])['patient_id'].count()
patient.groupby(['contacted_with', 'province'])['patient_id'].count()
confirmed_acc['date'] = pd.to_datetime(confirmed_acc['date'])
plt.figure(figsize=(20,20))

plt.title("Trend of Confirmed")

sns.lineplot(x="date", y="cases", data=confirmed_acc)
plt.figure(figsize=(20,10))

cases.plot('date',['acc_confirmed', 'acc_negative', 'acc_tested', 'acc_deceased'],figsize=(10,10), rot=30)

plt.title("Trend of cases by State")
import plotly.express as px

import plotly.offline as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from fbprophet import Prophet
data_confirmed_cases = cases[['date', 'acc_confirmed']]

data_confirmed_cases.dropna(inplace=True)

data_confirmed_cases['date'] = pd.to_datetime(data_confirmed_cases['date'])

data_confirmed_cases.tail()
date_deceased = pd.date_range('2020-01-21', '2020-03-01')

confirmed = {'date': date_deceased, 'acc_confirmed': 0}

fbp_confirmed = pd.DataFrame(data=confirmed)
data_confirmed_cases = data_confirmed_cases.append(fbp_confirmed)

data_confirmed_cases = data_confirmed_cases.sort_values(by='date').reset_index()

data_confirmed_cases.drop('index', axis=1, inplace=True)
data_confirmed_cases = data_confirmed_cases.rename(columns={"date": "ds", "acc_confirmed": "y"})

data_confirmed_cases['ds'] = pd.to_datetime(data_confirmed_cases['ds'])

data_confirmed_cases.head()
m = Prophet(

    changepoint_prior_scale=0.2,

    changepoint_range=0.95,

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



# for more information about changepoint take a look to this link: https://facebook.github.io/prophet/docs/trend_changepoints.html



m.fit(data_confirmed_cases)



future = m.make_future_dataframe(periods=15)

forecast_confirmed = m.predict(future)





forecast_confirmed[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
from fbprophet.plot import plot_plotly

fig = plot_plotly(m, forecast_confirmed)

py.iplot(fig) 
data_deceased_cases = cases[['date', 'acc_deceased']]

data_deceased_cases.dropna(inplace=True)

data_deceased_cases['date'] = pd.to_datetime(data_deceased_cases['date'])

data_deceased_cases.tail()
date_deceased = pd.date_range('2020-01-21', '2020-03-01')

deceased = {'date': date_deceased, 'acc_deceased': 0}

fbp_deceased = pd.DataFrame(data=deceased)
data_deceased_cases = data_deceased_cases.append(fbp_deceased)

data_deceased_cases = data_deceased_cases.sort_values(by='date').reset_index()

data_deceased_cases.drop('index', axis=1, inplace=True)
data_deceased_cases = data_deceased_cases.rename(columns={"date": "ds", "acc_deceased": "y"})

data_deceased_cases['ds'] = pd.to_datetime(data_deceased_cases['ds'])

data_deceased_cases.head()
m = Prophet(

    changepoint_prior_scale=0.2,

    changepoint_range=0.95,

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



# for more information about changepoint take a look to this link: https://facebook.github.io/prophet/docs/trend_changepoints.html



m.fit(data_deceased_cases)



future = m.make_future_dataframe(periods=15)

forecast_deceased = m.predict(future)





forecast_deceased[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
fig = plot_plotly(m, forecast_deceased)

py.iplot(fig) 


ex_stagnant_confirmed_cases = pd.date_range('2020-03-26', '2020-03-30')

stagnant_confirmed = {'date': ex_stagnant_confirmed_cases, 'acc_confirmed': 790}

fbp_stagnant_confirmed = pd.DataFrame(data=stagnant_confirmed)

data_confirmed_cases_staganant = cases.loc[20:23, ["date", "acc_confirmed"]] #only get from 22 mar 20 to 25 mar 20

data_confirmed_cases_staganant = data_confirmed_cases_staganant.append(fbp_stagnant_confirmed)

data_confirmed_cases_staganant['date'] = pd.to_datetime(data_confirmed_cases_staganant['date'])

data_confirmed_cases_staganant = data_confirmed_cases_staganant.sort_values(by='date').reset_index()

data_confirmed_cases_staganant.drop('index', axis=1, inplace=True)



plt.figure(figsize=(20,10))

sns.lineplot(x="date", y='acc_confirmed', data=data_confirmed_cases_staganant)

plt.title("Example of Stagnant Confirmed trend")