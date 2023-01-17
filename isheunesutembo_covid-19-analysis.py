import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from datetime import date, timedelta

from statsmodels.tsa.arima_model import ARIMA

from sklearn.cluster import KMeans

from fbprophet import Prophet

data1=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
data1.head()
data1.shape
data1.isna().sum()
data1.describe()
daily=data1.sort_values(['Date','Country/Region','Province/State'])

latest=data1[data1.Date==daily.Date.max()]

latest.head()
data=latest.rename(columns={"Country/Region":"country","Province/State":"state","Confirmed":"confirm","Deaths":"death","Recovered":"recover"})

data.head()
dgc=data.groupby("country")[['confirm','death','recover']].sum().reset_index()

dgc.head()
dgc.describe().T
import folium

worldmap=folium.Map(location=[32.4279,53.6880],zoom_start=4,tiles='Stamen Toner')



for Lat ,Long , state in zip(data['Lat'],data['Long'],data['state']):

    folium.CircleMarker([Lat,Long],

                       radius=5,

                       color='red',

                       popup=('State:'+str(state)+'<br>'),

                       fill_color="red",

                       fill_opacity=0.7).add_to(worldmap)

worldmap
fig=px.bar(dgc[['country','confirm']].sort_values('confirm',ascending=False),

          y="confirm",x="country",color='country',

          log_y=True,template='ggplot2',title='Confirmed Cases')

fig.show()
fig=px.bar(dgc[['country','recover']].sort_values('recover',ascending=False),

          y="recover",x="country",color="country",

          log_y=True,template='ggplot2',title='Recovered Cases')

fig.show()
fig=px.bar(dgc[['country','death']].sort_values('death',ascending=False),

          y="death",x="country",color="country",log_y=True,template='ggplot2',title="Death")

fig.show()
data1.head()
iran_data=data1[data1['Country/Region']=='Iran']

idata=iran_data.tail(22)

idata.head()
plt.figure(figsize=(50,15))

plt.bar(idata.Date, idata.Confirmed,label="Confirm")

plt.bar(idata.Date, idata.Recovered,label="Recovery")

plt.bar(idata.Date, idata.Deaths,label="Death")

plt.xlabel('Date')

plt.ylabel("Count")

plt.legend(frameon=True, fontsize=12)

plt.title("Confirmation vs Recoverey vs Death",fontsize=50)

plt.show()



f, ax = plt.subplots(figsize=(23,10))

ax=sns.scatterplot(x="Date", y="Confirmed", data=idata,

             color="black",label = "Confirm")

ax=sns.scatterplot(x="Date", y="Recovered", data=idata,

             color="red",label = "Recovery")

ax=sns.scatterplot(x="Date", y="Deaths", data=idata,

             color="blue",label = "Death")

plt.plot(idata.Date,idata.Confirmed,zorder=1,color="black")

plt.plot(idata.Date,idata.Recovered,zorder=1,color="red")

plt.plot(idata.Date,idata.Deaths,zorder=1,color="blue")
dgd=data.groupby("Date")[['confirm','death','recover']].sum().reset_index()

dgd.head()
r_cm=float(dgd.recover/dgd.confirm)

d_cm=float(dgd.death/dgd.confirm)
print("The percentage of recovery after confirmation is "+ str(r_cm*100) )

print("The percentage of death after confirmation is "+ str(d_cm*100) )
prophet=iran_data.iloc[: , [4,5 ]].copy() 

prophet.head()

prophet.columns = ['ds','y']

prophet.head()
m=Prophet()

m.fit(prophet)

future=m.make_future_dataframe(periods=365)

forecast=m.predict(future)

forecast
cnfrm = forecast.loc[:,['ds','trend']]

cnfrm = cnfrm[cnfrm['trend']>0]

cnfrm.head()

cnfrm=cnfrm.head(65)

cnfrm=cnfrm.tail(30)

cnfrm.columns = ['Date','Confirm']

cnfrm.head()
figure=plot_plotly(m,forecast)

py.iplot(figure)

figure=m.plot(forecast,xlabel="Date",ylabel="Confirmed Count")
figure=m.plot_components(forecast)
prophet_rec=iran_data.iloc[: , [4,7 ]].copy() 

prophet_rec.head()

prophet_rec.columns = ['ds','y']

prophet_rec.head()
m1=Prophet()

m1.fit(prophet_rec)

future_rec=m1.make_future_dataframe(periods=365)

forecast_rec=m1.predict(future_rec)

forecast_rec
rec = forecast_rec.loc[:,['ds','trend']]

rec = rec[rec['trend']>0]

rec.head()

rec=rec.head(65)

rec=rec.tail(30)

rec.columns = ['Date','Recovery']

rec.head()
figure_rec = plot_plotly(m1, forecast_rec)

py.iplot(figure_rec) 



figure_rec = m1.plot(forecast_rec,xlabel='Date',ylabel='Recovery Count')
prophet_dth=iran_data.iloc[: , [4,6 ]].copy() 

prophet_dth.head()

prophet_dth.columns = ['ds','y']

prophet_dth.head()
m2=Prophet()

m2.fit(prophet_dth)

future_dth=m2.make_future_dataframe(periods=365)

forecast_dth=m2.predict(future_dth)

forecast_dth
dth = forecast_dth.loc[:,['ds','trend']]

dth = dth[dth['trend']>0]

dth=dth.head(66)

dth=dth.tail(30)

dth.columns = ['Date','Death']

dth.head()
figure_dth = plot_plotly(m2, forecast_dth)

py.iplot(figure_dth) 



figure_dth = m2.plot(forecast_dth,xlabel='Date',ylabel='Death Count')
figure_dth=m2.plot_components(forecast_dth)
prediction = cnfrm

prediction['Recover'] = rec.Recovery

prediction['Death'] = dth.Death

prediction.head()
pr_pps = float(prediction.Recover.sum()/prediction.Confirm.sum())

pd_pps = float(prediction.Death.sum()/prediction.Confirm.sum())
print("The percentage of Predicted recovery after confirmation is "+ str(pr_pps*100) )

print("The percentage of Predicted Death after confirmation is "+ str(pd_pps*100) )
