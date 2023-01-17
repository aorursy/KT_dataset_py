import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
py.init_notebook_mode(connected=True)
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Active", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="natural earth",
                     animation_frame="Date",width=800, height=500,
                     color_continuous_scale='Reds',
                     range_color=[1000,100000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)
data_frame=pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
data_frame.head()
data_frame['Country/Region'].unique()
russia_frame = data_frame[data_frame['Country/Region'] ==  'Russia']
russia_frame.head()
confirmed_cases = russia_frame.Confirmed.max()
confirmed_cases
death_cases=russia_frame.Deaths.max()
death_cases
recovered_cases=russia_frame.Recovered.max()
recovered_cases
plt.figure(figsize=(50,40))
plt.bar(russia_frame.Date, russia_frame.Confirmed,label="Confirmed Cases")
plt.bar(russia_frame.Date, russia_frame.Recovered,label="Recovered Cases")
plt.bar(russia_frame.Date, russia_frame.Deaths,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(frameon=True, fontsize=42)
plt.title('ConfirmedCases vs RecoveredCases vs Deaths',fontsize=30)
plt.show()
f, ax = plt.subplots(figsize=(30,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=russia_frame,
             color="black",label = "ConfirmedCases")
ax=sns.scatterplot(x="Date", y="Recovered", data=russia_frame,
             color="red",label = "RecoveredCases")
ax=sns.scatterplot(x="Date", y="Deaths", data=russia_frame,
             color="blue",label = "Death")
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)

plt.legend(frameon=True, fontsize=22)
plt.plot(russia_frame.Date,russia_frame.Confirmed,zorder=1,color="black")
plt.plot(russia_frame.Date,russia_frame.Recovered,zorder=1,color="red")
plt.plot(russia_frame.Date,russia_frame.Deaths,zorder=1,color="blue")
import folium
map = folium.Map(location=[55.5852,95.2384 ], zoom_start=4,tiles='Open street map')

for lat, lon,Confirmed,Recovered,Deaths in zip(russia_frame['Latitude'], russia_frame['Longitude'],russia_frame['Confirmed'],russia_frame['Recovered'],russia_frame['Deaths']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                        
                      popup =(
                    'Confirmed: ' + str(Confirmed) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(map)
map
#russia_frame = russia_frame.tail(5)
confirmed_r = russia_frame.loc[:,['Date','Confirmed']]
confirmed_r.columns = ['ds','y']
confirmed_r.head()
m=Prophet()
m.fit(confirmed_r)
future=m.make_future_dataframe(periods=15)
forecast_confirmed_r=m.predict(future)
forecast_confirmed_r
confirmed_forecast = forecast_confirmed_r.loc[:,['ds','trend']]
confirmed_forecast = confirmed_forecast[confirmed_forecast['trend']>0]
confirmed_forecast.columns = ['Date','Confirmed']
confirmed_forecast
fig_r = plot_plotly(m, forecast_confirmed_r)
py.iplot(fig_r) 

fig_r = m.plot(forecast_confirmed_r,xlabel='Date',ylabel='Confirmation Count')
global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

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
plot_country_active_confirmed_recovered('Russia')