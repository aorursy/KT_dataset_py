# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fbprophet import Prophet
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
!pip install pmdarima
import pmdarima as pm
!pip install bubbly
!pip install opencage
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
df.shape
df.head()
df.columns
df.describe()
df['AvgTemperature'].value_counts()
print(df['AvgTemperature'].min())
print(df['AvgTemperature'].max())
df = df[df['AvgTemperature']>-99.0] 
sns.distplot(df['AvgTemperature'])
df['Region'].unique()
len(df['Region'].unique())
df['State'].value_counts().plot(kind= 'bar',figsize = (15,7))
df['State'].isna().sum()
(df['State'].isna().sum() / df['State'].shape[0])*100
len(df['City'].unique())
df.groupby('Region')['City'].nunique().plot(kind = 'bar')
df['Date'] = df['Day'].astype(str)+ "-" + df['Month'].astype(str) + "-" + df['Year'].astype(str)
df['Date'] = pd.to_datetime(df['Date'])
df.groupby('Year')['AvgTemperature'].mean().plot(figsize = (15,7))
yearly_region_data = df.groupby(['Region', 'Year']).mean().reset_index()
yearly_region_data
import plotly.express as px
fig = px.scatter(yearly_region_data, x="Year", y="AvgTemperature", color="Region")
fig.show()
from bubbly.bubbly import bubbleplot 
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

figure = bubbleplot(dataset=yearly_region_data, x_column='Year', y_column='AvgTemperature', 
    bubble_column='Region', time_column='Year', color_column='Region', 
    x_title="Year", y_title="Average Temperature", title='Avg Temperature over the years across Regions',
    scale_bubble=2.5, height=650)

iplot(figure, config={'scrollzoom': True})
plt.subplots(figsize=(15, 6))
sns.lineplot(x = 'Year', y = 'AvgTemperature', hue = 'Region', data = yearly_region_data)
plt.xticks(rotation = 90)
plt.legend()
plt.title('Temperature changes across regions from 1995-2020.')
plt.show()
df.boxplot(column='AvgTemperature', by='Region',figsize = (15,7))
df.groupby(['Region'])['AvgTemperature'].mean().plot(kind='bar',figsize=(17,7))
month_region_data = df.groupby(['Month', 'Year']).mean().reset_index().drop(columns = ['Year', 'Day'])
print(df['Date'].min())
print(df['Date'].max())
df['Year'].value_counts().plot(kind = 'bar',figsize=(17,7))
data = df[df['Date'].astype(str) < '2020-10-00 00:00:00']
from opencage.geocoder import OpenCageGeocode
key = '737b68f4ab6741308ac6cb4b0d35ee8e'
geocoder = OpenCageGeocode(key)
cities = data.groupby('City')['AvgTemperature'].mean().reset_index()
Latitude = []
Longitude = []
for i in list(cities['City']):
    #print(i)
    try:
        result = geocoder.geocode(i, no_annotations=1, language='es')
        Latitude.append(list(result[0].get('geometry').values())[0])
        Longitude.append(list(result[0].get('geometry').values())[1])
    except:
        Latitude.append(-34.901112)
        Longitude.append(-56.164532)
cities['Latitude'] = Latitude
cities['Longitude'] = Longitude
cities
import folium 
import webbrowser
latitude = 37.0902
longitude = -95.7129
maps = folium.Map(location=[latitude, longitude], zoom_start=5)
for lat, lon, temp, city in zip(cities['Latitude'], cities['Longitude'], cities['AvgTemperature'], cities['City']):
    folium.CircleMarker(
        [lat, lon],
        radius=0.1*temp,
        popup = ('City: ' + str(city).capitalize() + '<br>'
                 'Temerature: ' + str(temp) + '<br>'
                 +'%'
                ),
        color='b',
        key_on = city,
        threshold_scale=[0,1,2,3],
        fill=True,
        fill_opacity=0.7
        ).add_to(maps)

maps

from fbprophet import Prophet
from fbprophet.plot import plot_yearly
for region in list(set(df['City'])):
    temp = df[df['City'] == region]
    temp = temp.sort_values('Date')
    print(temp[['Date','AvgTemperature']])
    plt.plot(temp['Date'],temp['AvgTemperature'])
    plt.show()
    temp.rename(columns = {'Date':'ds','AvgTemperature':'y'},inplace = True)
    model = Prophet()
    
    m = Prophet().fit(temp)
    a = plot_yearly(m)
            
    model.fit(temp[['ds','y']])
    future = model.make_future_dataframe(periods=100,freq='D')
    forecast = model.predict(future)
    model.plot(forecast)
    
    break
for region in list(set(df['City'])):
    temp = df[df['City'] == region]
    temp = temp.sort_values('Date')
    print(temp[['Date','AvgTemperature']])
    ARIMA_model = pm.auto_arima(temp['AvgTemperature'], start_p=1, d= None , start_q=1, max_p=6, max_d=6, max_q=6, start_P=0,D = 1, start_Q=0, max_P=5,
                             max_D=5, max_Q=0, max_order=None, m=12, seasonal=True, stationary=False, information_criterion='bic', alpha=0.01, test='kpss', 
                              seasonal_test='ch', stepwise=True, n_jobs=1)
    break
ARIMA_model.summary()
forecast = ARIMA_model.predict(n_periods=12)
forecast_start = '2021-01-01'
forecast_end = '2021-12-01'
month_list = [i.strftime("%Y-%m-%d") for i in pd.date_range(start=forecast_start, end=forecast_end, freq='MS')]
df_forecast = pd.DataFrame()
df_forecast['Date'] = month_list
df_forecast['Forecasted_Temperature'] = forecast