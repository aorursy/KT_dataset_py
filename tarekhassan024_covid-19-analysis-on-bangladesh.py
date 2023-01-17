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
from sklearn.cluster import KMeans
from fbprophet import Prophet
dataset=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")

dataset.shape

dataset.describe()

daily = dataset.sort_values(['Date','Country/Region','Province/State'])
latest = dataset[dataset.Date == daily.Date.max()]
latest.head()
data=latest.rename(columns={ "Country/Region": "country", "Province/State": "state","Confirmed":"confirm","Deaths": "death","Recovered":"recover"})
data.head()
dgc=data.groupby("country")[['confirm', 'death', 'recover']].sum().reset_index()

dgc.head()
import folium
worldmap = folium.Map(location=[23.6850,90.3563 ], zoom_start=5,tiles='Stamen Toner')

for Lat, Long, state in zip(data['Lat'], data['Long'],data['state']):
    folium.CircleMarker([Lat, Long],
                        radius=5,
                        color='red',
                      popup =('State: ' + str(state) + '<br>'),
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(worldmap)
worldmap
fig = px.bar(dgc[['country', 'confirm']].sort_values('confirm', ascending=False), 
             y="confirm", x="country", color='country', 
             log_y=True, template='ggplot2', title='Confirmed Cases')
fig.show()
fig = px.bar(dgc[['country', 'recover']].sort_values('recover', ascending=False), 
             y="recover", x="country", color='country', 
             log_y=True, template='ggplot2', title='Recovered Cases')
fig.show()
fig = px.bar(dgc[['country', 'death']].sort_values('death', ascending=False), 
             y="death", x="country", color='country', 
             log_y=True, template='ggplot2', title='Death')
fig.show()
bd_data = dataset[dataset['Country/Region']=='Bangladesh']
bdata = bd_data.tail(22)
bdata.tail()
plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Confirmed,label="Confirm")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation",fontsize=50)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Recovered,label="Recovery")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Recoverey",fontsize=50)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Death",fontsize=50)
plt.show()
plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Confirmed,label="Confirm")
plt.bar(bdata.Date, bdata.Recovered,label="Recovery")
plt.bar(bdata.Date, bdata.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death",fontsize=50)
plt.show()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=bdata,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered", data=bdata,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=bdata,
             color="blue",label = "Death")
plt.plot(bdata.Date,bdata.Confirmed,zorder=1,color="black")
plt.plot(bdata.Date,bdata.Recovered,zorder=1,color="red")
plt.plot(bdata.Date,bdata.Deaths,zorder=1,color="blue")
bdata['Confirmed_new'] = bdata['Confirmed']-bdata['Confirmed'].shift(1)
bdata['Recovered_new'] = bdata['Recovered']-bdata['Recovered'].shift(1)
bdata['Deaths_new'] = bdata['Deaths']-bdata['Deaths'].shift(1)
bdata = bdata.fillna(0)

plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Confirmed_new,label="Confirm Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed Cases',fontsize = 35)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Recovered_new,label="Recovered Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Recovered Cases',fontsize = 35)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(bdata.Date, bdata.Deaths_new,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Deaths',fontsize = 35)
plt.show()
bdata.head()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="Confirmed_new", data=bdata,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered_new", data=bdata,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths_new", data=bdata,
             color="blue",label = "Death")
plt.plot(bdata.Date,bdata.Confirmed_new,zorder=1,color="black")
plt.plot(bdata.Date,bdata.Recovered_new,zorder=1,color="red")
plt.plot(bdata.Date,bdata.Deaths_new,zorder=1,color="blue")
dgd=data.groupby("Date")[['confirm', 'death', 'recover']].sum().reset_index()

dgd.head()
r_cm = float(dgd.recover/dgd.confirm)
d_cm = float(dgd.death/dgd.confirm)
print("The percentage of recovery after confirmation is "+ str(r_cm*100) )
print("The percentage of death after confirmation is "+ str(d_cm*100) )
global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
# This functions smooths data, thanks to Dan Pearson. We will use it to smooth the data for growth factor.
def smoother(inputdata,w,imax):
    data = 1.0*inputdata
    data = data.replace(np.nan,1)
    data = data.replace(np.inf,1)
    #print(data)
    smoothed = 1.0*data
    normalization = 1
    for i in range(-imax,imax+1):
        if i==0:
            continue
        smoothed += (w**abs(i))*data.shift(i,axis=0)
        normalization += w**abs(i)
    smoothed /= normalization
    return smoothed

def growth_factor(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    confirmed_iminus2 = confirmed.shift(2, axis=0)
    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)

def growth_ratio(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    return (confirmed/confirmed_iminus1)

# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.
def plot_country_active_confirmed_recovered(country):
    
    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.
    country_data = global_data[global_data['Country/Region']==country]
    table = country_data.drop(['SNo','Province/State', 'Last Update'], axis=1)
    table['ActiveCases'] = table['Confirmed'] - table['Recovered'] - table['Deaths']
    table2 = pd.pivot_table(table, values=['ActiveCases','Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)
    table3 = table2.drop(['Deaths'], axis=1)
   
    # Growth Factor
    w = 0.5
    table2['GrowthFactor'] = growth_factor(table2['Confirmed'])
    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)

    # 2nd Derivative
    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative
    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)


    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio
    table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])
    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)
    
    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.
    table2['GrowthRate']=np.gradient(np.log(table2['Confirmed']))
    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)
    
    # horizontal line at growth rate 1.0 for reference
    x_coordinates = [1, 100]
    y_coordinates = [1, 1]
    #plots
    table2['Deaths'].plot(title='Deaths')
    plt.show()
    table3.plot() 
    plt.show()
    table2['GrowthFactor'].plot(title='Growth Factor')
    plt.plot(x_coordinates, y_coordinates) 
    plt.show()
    table2['2nd_Derivative'].plot(title='2nd_Derivative')
    plt.show()
    table2['GrowthRatio'].plot(title='Growth Ratio')
    plt.plot(x_coordinates, y_coordinates)
    plt.show()
    table2['GrowthRate'].plot(title='Growth Rate')
    plt.show()

    return 
plot_country_active_confirmed_recovered('Bangladesh')

prophet=bd_data.iloc[: , [4,5 ]].copy() 
prophet.head()
prophet.columns = ['ds','y']
prophet.head()
m=Prophet()
m.fit(prophet)
future=m.make_future_dataframe(periods=15)
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm']
cnfrm.head()
figure = plot_plotly(m, forecast)
py.iplot(figure) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
figure=m.plot_components(forecast)

prophet_rec=bd_data.iloc[: , [4,7 ]].copy() 
prophet_rec.head()
prophet_rec.columns = ['ds','y']
prophet_rec.head()
m1=Prophet()
m1.fit(prophet_rec)
future_rec=m1.make_future_dataframe(periods=15)
forecast_rec=m1.predict(future_rec)
forecast_rec
rec = forecast_rec.loc[:,['ds','trend']]
rec = rec[rec['trend']>0]
rec=rec.tail(15)
rec.columns = ['Date','Recovery']
rec.head()
figure_rec = plot_plotly(m1, forecast_rec)
py.iplot(figure_rec) 

figure_rec = m1.plot(forecast_rec,xlabel='Date',ylabel='Recovery Count')
figure_rec=m1.plot_components(forecast_rec)

prophet_dth=bd_data.iloc[: , [4,6 ]].copy() 
prophet_dth.head()
prophet_dth.columns = ['ds','y']
prophet_dth.head()
m2=Prophet()
m2.fit(prophet_dth)
future_dth=m2.make_future_dataframe(periods=20)
forecast_dth=m2.predict(future_dth)
forecast_dth
dth = forecast_dth.loc[:,['ds','trend']]
dth = dth[dth['trend']>0]
dth=dth.tail(15)
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
