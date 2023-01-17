import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

from datetime import date 

import datetime



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



from statsmodels.tsa.arima_model import ARIMA
#US Cases

us_cases = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')



#India Cases

case_time_series=pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
case_time_series['Date']=pd.to_datetime(case_time_series['Date']+'20', format='%d %B %y')

case_time_series.info()
temp=us_cases.groupby('date')

us_cases_final=temp[['cases','date']].sum()

us_cases_final.reset_index(level=0, inplace=True)

us_cases_final['date']=pd.to_datetime(us_cases_final['date'])
us_cases.head()
plt.figure(figsize=(8,8))

plt.plot(us_cases_final.index,us_cases_final['cases'], label = 'US')

plt.plot(case_time_series.index,'Total Confirmed',data=case_time_series,label = 'India')

plt.title('Number of Coronavirus Cases')

plt.xlabel('Days')

plt.ylabel('Number of cases')

plt.legend()

plt.grid(zorder = 30)

plt.show()
data=case_time_series[['Date','Daily Confirmed','Total Confirmed']]

#confirm_cs = pd.DataFrame(data).cumsum()

arima_data = data.reset_index()

arima_data.columns = ['index','Date','Daily Confirmed','Total Confirmed']
start_date = case_time_series['Date'].max()

prediction_dates_india = []

for i in range(150):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates_india.append(date)

    start_date = date
model = ARIMA(arima_data['Total Confirmed'].values, order=(6,2,1))

fit_model = model.fit(trend='c', full_output=True, disp=True)

forecast=fit_model.forecast(steps=150)

pred = list(forecast[0])



#fit_model.summary()
data=us_cases_final[['date','cases']]

#confirm_cs = pd.DataFrame(data).cumsum()

arima_data = data.reset_index()

arima_data.columns = ['index','date','cases']
model = ARIMA(arima_data['cases'].values, order=(6,2,1))

fit_model = model.fit(trend='c', full_output=True, disp=True)

forecast=fit_model.forecast(steps=150)

pred1 = list(forecast[0])



#fit_model.summary()
start_date = us_cases_final['date'].max()

prediction_dates = []

for i in range(150):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

            

fig = go.Figure()



fig = px.line(case_time_series, x=case_time_series['Date'], y=case_time_series['Total Confirmed'],title="India Total Cases", template="plotly_dark")



fig.add_scatter(y=case_time_series['Total Confirmed'],x=case_time_series['Date'],name="India Total Cases")

fig.add_scatter(y=pred,x=prediction_dates_india,name="India Predicted Total Cases")

fig.add_scatter(y=us_cases_final['cases'],x=us_cases_final['date'],name="US Total Cases")

fig.add_scatter(y=pred1,x=prediction_dates, name="US Predicted Total Cases")



fig.update_xaxes(

    #rangeslider_visible=True,

    rangeselector=dict(

        bgcolor="#00CED1",

        buttons=list([

            dict(count=1, label="Last 1 Month", step="month", stepmode="backward"),

            dict(count=3, label="Last 3 month", step="month", stepmode="backward"),

            dict(count=6, label="Last 6 Month", step="month", stepmode="backward"),

            dict(step="all")

        ])

    )

)





fig.show()