import pandas as pd

import numpy as np



from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

from plotly.offline import iplot, init_notebook_mode



import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



import seaborn as sns

import scipy as sp



import matplotlib.pyplot as plt

import matplotlib.dates as mdates



daily_report =pd.read_csv('../input/korea-corona/korea_corona_confirmed_by_region.csv')

col= ['date', 'Seoul','Busan','Incheon','Gwangju' ,'Daejeon', 'Ulsan','Sejong', 'Gyeonggi-do', 'Gangwon-do', 'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do', 'Jeollanam-do',

      'Gyeongsangnam-do','Jeju-do', 'Daegu','Gyeongsangbuk-do','total']

daily_report.columns=col

daily_report.info()
daily_report
import datetime

end = datetime.datetime.now() - datetime.timedelta(1)

date_index = pd.date_range('2020-01-23', end)

daily_report.index = date_index

daily_report.drop(['date'], axis=1, inplace=True)

daily_report.info()
import cufflinks as cf 

cf.go_offline(connected=True)

daily_report.iloc[:,-1].iplot( kind='bar') # total count
latest = daily_report.iloc[-1,:17]

latest = pd.DataFrame(latest)
import plotly.express as px

index_name= ['Seoul','Busan','Incheon','Gwangju' ,'Daejeon', 'Ulsan','Sejong', 'Gyeonggi-do', 'Gangwon-do', 'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do', 'Jeollanam-do',

      'Gyeongsangnam-do','Jeju-do', 'Daegu','Gyeongsangbuk-do']

fig = px.pie(latest, values=latest.values, names=index_name)

fig.show()
daily_by_2_18 = daily_report.iloc[25:,:-1]

daily_by_2_18.iplot(kind='bar')
daily_by_Gyeongbuk_Daegu = daily_report.iloc[23:,15:]

daily_by_Gyeongbuk_Daegu.iplot(kind='bar')
daily_by_Daegu = daily_report.iloc[23:,15]

daily_by_Daegu = pd.DataFrame(daily_by_Daegu)

daily_by_Daegu.reset_index(inplace=True)

df_prophet = daily_by_Daegu.rename(columns={ 'index': 'ds', 'Daegu': 'y' })



df_prophet
m = Prophet(

    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

    changepoint_range=0.95, # place potential changepoints in the first 98% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



m.fit(df_prophet)



future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)





forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
fig = plot_plotly(m, forecast)

fig.update_layout(title_text="Accumulated Confirmed_patient of nCOV-19 in Daegu")

# 대구는 아직까지는 확진자가 미비하게 증가할것으로 보여집니다

py.iplot(fig) 

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
sns.distplot(m.params["delta"], kde=False, fit=sp.stats.laplace)

plt.box(False)
daily_by_Gyeongbuk = daily_report.iloc[23:,16]

daily_by_Gyeongbuk = pd.DataFrame(daily_by_Gyeongbuk)



daily_by_Gyeongbuk.reset_index(inplace=True)

df_prophet = daily_by_Gyeongbuk.rename(columns={ 'index': 'ds', 'Gyeongsangbuk-do': 'y' })
m = Prophet(

    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible

    changepoint_range=0.95, # place potential changepoints in the first 98% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



m.fit(df_prophet)



future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)





forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
fig = plot_plotly(m, forecast)

fig.update_layout(title_text="Accumulated Confirmed_patient of nCOV-19 in Gyeongbuk")

py.iplot(fig) 

sns.distplot(m.params["delta"], kde=False, fit=sp.stats.laplace)

plt.box(False)
import cufflinks as cf 

cf.go_offline(connected=True)

daily_report.iloc[:, 15:].iplot( fill=True) # total count
total = daily_report.iloc[:, 17]

print(total.count())





increased_daily =[0,]

for i in date_index:

  increased_d = total[i+1] - total[i]

  increased_daily.append(increased_d) 
total= pd.DataFrame(total)

total['increased_daily']= increased_daily

total
total.iloc[:,-1].iplot(kind='bar')
total_by_3_3 = total.iloc[40:50, :]

total_by_3_3['active'] = [35555,28414,21810,21832,19620,19376,17458,18452,18540,17727] # 10일간 본 검사중인 사람수 누적결과 by CDC

total_by_3_3
total_by_3_3['active'].iplot(kind='bar')