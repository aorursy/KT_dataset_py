import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math as m

from pandas import read_csv



from datetime import datetime



import os

import glob



import rasterio as rio

import folium 



import geopandas

import tifffile as tiff

from folium import plugins

from shapely.geometry import Point

import rasterstats

from rasterstats import zonal_stats, point_query
pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)

import warnings

warnings.filterwarnings('ignore')

# Code for displaying plotly express plot

def configure_plotly_browser_state():

  import IPython

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',

            },

          });

        </script>

        '''))

  
from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.express as px

configure_plotly_browser_state()
from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
data_path = "/kaggle/input/ds4g-environmental-insights-explorer"



power_plants_df = pd.read_csv("/kaggle/input/ds4g-eie-no2emission-eda/power_plants.csv")

power_plants_fossil_df = pd.read_csv("/kaggle/input/ds4g-eie-no2emission-eda/fossil.csv")



no2_dfs_gldas_df = pd.read_csv("/kaggle/input/ds4g-eie-no2emission-eda/no2_dfs_gldas_df.csv")

power_plants_renew_df = pd.read_csv("/kaggle/input/ds4g-eie-no2emission-eda/renew.csv")

#pp_gdf = pd.read_csv("/kaggle/input/kaggle-ds4g-part2/pp_gdf.csv")
pp_gdf = geopandas.GeoDataFrame(

    power_plants_df, geometry=geopandas.points_from_xy(power_plants_df.longitude, power_plants_df.latitude))



world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))



# We restrict to Puerto Rica.

ax = world[world.continent == 'Puerto Rica'].plot(

    color='white', edgecolor='black')



# We can now plot our ``GeoDataFrame``.

pp_gdf.plot(ax=ax, color='magenta')



plt.show()

pp_gdf['name'] = pp_gdf['name'] + ':' + pp_gdf['primary_fuel']
#@title

'''

import ee

from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("akshi")

'''
#@title

'''

ee.Authenticate()

!cat ~/.config/earthengine/credentials

'''
#@title

'''

#user_secret = "" # Your user secret, defined in the add-on menu of the notebook editor

#refresh_token = UserSecretsClient().get_secret(user_secret)

credentials = Credentials(

        None,

        refresh_token=secret_value_0,

        token_uri=ee.oauth.TOKEN_URI,

        client_id=ee.oauth.CLIENT_ID,

        client_secret=ee.oauth.CLIENT_SECRET,

        scopes=ee.oauth.SCOPES)

ee.Initialize()

#ee.Initialize(credentials=credentials)

'''
def initMap(df, lat, lon):

    location = [lat, lon]

    Map = folium.Map(location=location, zoom_start=8)

    return Map

pp_geo_df_ = pp_gdf.copy()

pp_geo_df_['geometry'] = pp_gdf.geometry.buffer(0.05)
lat=18.140178; lon=-66.664513 #puerto rico

Map = initMap(power_plants_df, lat, lon)   

#plot_points_on_map(power_plants,lat,lon,10)

for i in range(pp_geo_df_.shape[0]):

    folium.GeoJson(pp_geo_df_.geometry[i]).add_to(Map)



Map
def img_stat(pp_geometry, name, primary_fuel, j):

  global res , pp_monthwise



  stats = []

  dates = []

  #s5p_no2_20180701T161259_20180707T175356

  s5p_no2_path = glob.glob(data_path+'/eie_data/s5p_no2/*')

  for (i, image) in enumerate(s5p_no2_path):

      date = datetime.strptime(image[76:84], '%Y%m%d')

      stat = rasterstats.zonal_stats(pp_geometry,

                                    image,

                                    band=2, #2: tropospheric_NO2_column_number_density

                                    stats=['mean'])

      stat = stat[0] # get location of pp

      stat = stat['mean'] # retrieve stat

      dates.append(date)

      stats.append(stat)

      

  results = pd.DataFrame(index=dates, data=stats, columns=[primary_fuel])

  results.plot()

  plt.title('tropospheric NO2_column_number_density over time period\n')   

  # chking -------

  pp_all_month = pd.DataFrame(list(zip(dates, stats)), columns= ['Date','no2-mean'], index=None)

  pp_all_month['pp'] = name

  if j==0 :

    res = pp_all_month

  else:

    res = res.append(pp_all_month, ignore_index= True)  

     

  # perform GroupBy operation over monthly frequency

  res.Date = pd.to_datetime(res.Date,format='%Y-%m-%d')

  a = res.set_index('Date').groupby(pd.Grouper(freq='M'))['no2-mean'].sum().reset_index()

  a['pp'] = name

  if j==0 :

    pp_monthwise = a

  else:

    pp_monthwise = pp_monthwise.append(a, ignore_index = True )



  # chking -------

  x = results.sum()

  #print (x)

  new_row = {"plant_name": name, "no2_emission": x}

  emission_df.loc[j] = new_row



  return pp_monthwise   
emission_df = pd.DataFrame()



#pp_all_month = pd.DataFrame(columns= ['Date','no2-mean','pp'])



emission_df['plant_name'] =""

emission_df['no2_emission'] = 0.0

for i in range(0,pp_gdf.shape[0]):

  img_stat(pp_gdf.geometry[i], pp_gdf.name[i], pp_gdf.primary_fuel[i], i )



emission_df.info()

emission_df['plant_name'] = emission_df['plant_name'].astype(str)

emission_df['no2_emission'] = emission_df['no2_emission'].astype(float)



pp_monthwise.info()

pp_monthwise['Date'] = pd.to_datetime(pp_monthwise['Date'])

pp_monthwise['no2-mean'] = pp_monthwise['no2-mean'].astype(float)

pp_monthwise['pp'] = pp_monthwise['pp'].astype(str)



pp_monthwise.head()
emission_df.to_csv('emission_df.csv')

pp_monthwise.to_csv('pp_monthwise.csv')
'''

plt.figure(figsize=(40,20))  

fig, ax = plt.subplots()    

width = 0.5 # the width of the bars 

ind = np.arange(len(emission_df['no2_emission']))  # the x locations for the groups

ax.barh(ind, emission_df['no2_emission'], width, color="blue")

ax.set_yticks(ind+width/2)

ax.set_yticklabels(emission_df['plant_name'], minor=False)

plt.title('NO2 Emission')

plt.xlabel('no2 emission')

plt.ylabel('plant name')      

#plt.show()

plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictur

'''
from matplotlib import style

style.use('ggplot')

plt.figure(figsize=(12,8))                                 # Setting the figure size

sns.distplot(emission_df['no2_emission'])                     # Creating the histogram

plt.show()
import plotly.graph_objects as go

configure_plotly_browser_state()

fig = go.Figure(data=[go.Bar(

            x=emission_df['plant_name'], y=emission_df['no2_emission'],

            text="",

            textposition='auto'

        )])



fig.show()

#print ("All PR Power plants NO2 emission\n")
power_plants_df = power_plants_df.sort_values('capacity_mw',ascending=False).reset_index()

power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']][29:30]

quantity_of_electricity_generated = power_plants_df['estimated_generation_gwh'][29:30].values

print('Quanity of Electricity Generated: ', quantity_of_electricity_generated)
power_plants_fossil_df[power_plants_fossil_df.primary_fuel=='Coal']

annual_EF_coal = power_plants_fossil_df.loc[power_plants_fossil_df['primary_fuel'] == 'Coal', 'no2_emission_in_mols'].values

print ("Emission factor for Coal PP A.E.S. Corp {:.8f} mol * h / m^2 * gw".format(float(annual_EF_coal/((365*24)))))
power_plants_fossil_df[power_plants_fossil_df.primary_fuel=='Oil']

annual_EF_oil = power_plants_fossil_df.loc[power_plants_fossil_df['primary_fuel'] == 'Oil', 'no2_emission_in_mols'].values

print ("Emission factor for Vieques EPP {:.8f} mol * h / m^2 * gw".format(float(annual_EF_oil[3]/(365*24))))
power_plants_fossil_df['no2_emission(mol / hr)'] = power_plants_fossil_df['no2_emission_in_mols'] / (365*24)
power_plants_fossil_df[['capacity_mw','estimated_generation_gwh','name','primary_fuel','input_energy_Btu',

                        #'capacity_utilization',

                        'Efficiency_percent',

                        'no2_Emission_for_input_Btu_in_gms','no2_emission_in_tons', 'no2_emission_in_mols', 'no2_emission(mol / hr)']]
power_plants_fossil_df['EF(mol * h / m^2 * gw)'] = power_plants_fossil_df['no2_emission(mol / hr)'] / power_plants_fossil_df['estimated_generation_gwh'] 

power_plants_fossil_df['EF_daily'] = power_plants_fossil_df['no2_emission(mol / hr)'] / power_plants_fossil_df['estimated_generation_gwh'] * 24

power_plants_fossil_df['EF_yearly'] = power_plants_fossil_df['no2_emission(mol / hr)'] / power_plants_fossil_df['estimated_generation_gwh'] * (24 *365)
power_plants_fossil_df[['capacity_mw','estimated_generation_gwh','name','primary_fuel','input_energy_Btu',#'capacity_utilization',

                        'Efficiency_percent','no2_Emission_for_input_Btu_in_gms','no2_emission_in_tons', 'no2_emission_in_mols', 'no2_emission(mol / hr)','EF(mol * h / m^2 * gw)','EF_daily','EF_yearly']]
power_plants_fossil_df[['capacity_mw','estimated_generation_gwh','name','primary_fuel','input_energy_Btu',#'capacity_utilization',

                        'Efficiency_percent','no2_Emission_for_input_Btu_in_gms','no2_emission_in_tons', 'no2_emission_in_mols', 'no2_emission(mol / hr)','EF(mol * h / m^2 * gw)',

                        'EF_daily','EF_yearly']]
MEF_df = power_plants_fossil_df

power_plants_fossil_df['plant_name'] = power_plants_fossil_df['name'] + ':' + power_plants_fossil_df['primary_fuel']

power_plants_fossil_df = pd.merge(power_plants_fossil_df, emission_df, left_on=('plant_name'), right_on='plant_name')
power_plants_fossil_df['MEF'] = power_plants_fossil_df['no2_emission'] / (power_plants_fossil_df['Efficiency_percent'] / 100)
#power_plants_fossil_df.head()

print ('Marginal Emission Factor : ', power_plants_fossil_df['MEF'].sum())

print ('Average No2 Emission : ', power_plants_fossil_df['no2_emission'].sum())
power_plants_fossil_df[['capacity_mw','estimated_generation_gwh','name','primary_fuel','input_energy_Btu','capacity_utilization',

                        'Efficiency_percent','no2_Emission_for_input_Btu_in_gms','no2_emission_in_tons', 'no2_emission_in_mols', 'no2_emission(mol / hr)','EF(mol * h / m^2 * gw)',

                        'EF_daily','EF_yearly','MEF']]
power_plants_fossil_df.to_csv('power_plants_fossil_df.csv')
no2_dfs_gldas_df.date_from = pd.to_datetime(no2_dfs_gldas_df.date_from)

no2_weather_values_stats = no2_dfs_gldas_df.groupby('date_from').mean().reset_index()

print('We have data for {} days'.format(no2_dfs_gldas_df['date_from'].nunique()))
train= pd.DataFrame()

train['ds'] = pd.to_datetime(no2_weather_values_stats["date_from"])

train['y']=no2_weather_values_stats["tropospheric_NO2"]

indexedData = train.set_index('ds')

indexedData.head()
moving_average = indexedData.rolling(window=12).mean()

plt.plot(indexedData, color='blue', label='Original')

plt.plot(moving_average, color='red', label='Rolling Mean')

plt.legend(loc='best')

plt.title('Moving Average-Annual')
# The 'MS' string groups the data in buckets by start of the month

#train.set_index('y', inplace=True)

y = indexedData['y'].resample('MS').mean()



# The term bfill means that we use the value before filling in missing values

y = y.fillna(y.bfill())



#print(y)
import itertools

import statsmodels.api as sm

pd.plotting.register_matplotlib_converters() # Add this 

plt.style.use('fivethirtyeight')

f, ax = plt.subplots(figsize=(14,8))

ax.set_xlabel('date_from', fontsize=15)

ax.set_ylabel('N02 values', fontsize=15);

indexedData['y'].plot(figsize=(15, 6))

plt.show()
# Define the p, d and q parameters to take any value between 0 and 2

p = d = q = range(0, 2)



# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
warnings.filterwarnings("ignore") # specify to ignore warning messages



for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(indexedData['y'],

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)



            results = mod.fit()



            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(indexedData['y'],

                                order=(4, 2, 1),

                                seasonal_order=(1, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)



results = mod.fit()



print(results.summary())
results.plot_diagnostics(figsize=(10, 8))

#plt.show()

pred = results.get_prediction(start=270,end =330, dynamic=False)

pred_ci = pred.conf_int()

pred_ci.head()
ax = y['2018':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='r', alpha=.2)



ax.set_xlabel('Date')

ax.set_ylabel('NO2 Levels')

plt.legend()



plt.show()
mte_forecast = pred.predicted_mean

#print (indexedData['2019-03-05':])

mte_truth = indexedData['2019-03-05':]





# Compute the mean square error

mse = ((mte_forecast - mte_truth) ** 2).mean()

print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))

#print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'.format(np.sqrt(sum((mte_forecast-mte_truth)**2)/len(mte_forecast))))
from fbprophet import Prophet
m = Prophet()

m.fit(train)

future = m.make_future_dataframe(periods=100)

forecast = m.predict(future)

forecast
m.plot_components(forecast)
import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()
py.iplot([

    go.Scatter(x=train['ds'], y=train['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')

])
m = Prophet(changepoint_prior_scale=2.5)

m.fit(train)

future = m.make_future_dataframe(periods=180)

forecast = m.predict(future)
# Calculate root mean squared error.

print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:100, 'yhat']-train['y'])**2)) )

py.iplot([

    go.Scatter(x=train['ds'], y=train['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')

])
def overlay_image_on_puerto_rico_df(df, img, zoom):

    lat_map=df.iloc[[0]].loc[:,["latitude"]].iat[0,0]

    lon_map=df.iloc[[0]].loc[:,["longitude"]].iat[0,0]

    m = folium.Map([lat_map, lon_map], zoom_start=zoom)

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    folium.raster_layers.ImageOverlay(

        image=img,

        bounds = [[18.56,-67.32,],[17.90,-65.194]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df["latitude"].iloc[i],df["longitude"].iloc[i]],

                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(m)

        

    return m
image = data_path + '/eie_data/s5p_no2/s5p_no2_20190510T164338_20190516T183223.tif'

#overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):

overlay_image_on_puerto_rico_df(power_plants_df,image,zoom=8)
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
lon = []

lat = []

NO2 = []

imp_img = np.flip(tiff.imread(image))

for i in range(imp_img[:,:,1].shape[0]):

    for j in range(imp_img[:,:,1].shape[1]):

        #print(imp_img[:,:,1][i,j])

        NO2.append(imp_img[:,:,1][i,j])

        lon.append(i)

        lat.append(j)

        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)
results = pd.DataFrame(columns=['NO2', 'lat', 'lon'])

results = pd.DataFrame({'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon)})
data_scaled = scaler.fit_transform(results)

pred = KMeans(n_clusters=15).fit_predict(results)
plt.figure()

pred_image = pred.reshape(148, 475)

sns.heatmap(pred_image)
Error =[]

for i in range(1, 15):

    kmeans = KMeans(n_clusters = i).fit(results)

    kmeans.fit(results)

    Error.append(kmeans.inertia_)



plt.plot(range(1, 15), Error)

plt.title('Elbow method')

plt.xlabel('No of clusters')

plt.ylabel('Error')

plt.show()

kmeans.cluster_centers_
overlay_image_on_puerto_rico_df(power_plants_df,pred_image,zoom=8)
gldas_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/*')

gldas_files = sorted(gldas_files)

gfs_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*')

gfs_files = sorted(gfs_files)
#(gldas_files[2280:2329])

#gfs_files[1140:1165]
gldas_files_daily = []

for i in range(0,len(gldas_files[2280:2329]),8):

    #print(gldas_files[i:i+8])

    gldas_files_daily.append(gldas_files[i:i+8])



gfs_files_daily = []

for i in range(0,len(gfs_files[1140:1165]),4):

    #print(gfs_files[i:i+4])

    gfs_files_daily.append(gfs_files[i:i+4])
image_reglession_u = []

image_reglession_v = []

image_reglession_t = []

image_reglession_sh = []

image_reglession_rh = []

image_reglession_tp = []

image_reglession_rn = []



for i in range(len(gfs_files_daily)):

    gfs_tmp = gfs_files_daily[i]

    gldas_tmp = gldas_files_daily[i]

    array_wind_u = []

    array_wind_v = []

    array_t = []

    array_sh = []

    array_rh = []

    array_tp = []

    array_rn = []

    for j in range(len(gfs_tmp)):

        gfs_image_t = tiff.imread(gfs_tmp[j])[:,:,0]

        gfs_image_sh = tiff.imread(gfs_tmp[j])[:,:,1]

        gfs_image_rh = tiff.imread(gfs_tmp[j])[:,:,2]

        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]

        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]

        gfs_image_tp = tiff.imread(gfs_tmp[j])[:,:,5]        

        gldas_image_rn = tiff.imread(gldas_tmp[2*j])[:,:,7]

        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]

        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]



        #fill na by mean

        gfs_image_t = np.nan_to_num(gfs_image_t, nan=np.nanmean(gfs_image_t))

        gfs_image_sh = np.nan_to_num(gfs_image_sh, nan=np.nanmean(gfs_image_sh))

        gfs_image_rh= np.nan_to_num(gfs_image_rh, nan=np.nanmean(gfs_image_rh))

        gfs_image_tp = np.nan_to_num(gfs_image_tp, nan=np.nanmean(gfs_image_tp))

        

        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))

        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))

        

        gldas_image_rn = np.nan_to_num(gldas_image_rn, nan=np.nanmean(gldas_image_rn))

        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))

        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))

       

        gldas_image = (gldas_image1 + gldas_image2)/2

        wind_u = gfs_image_u * gldas_image

        wind_v = gfs_image_v * gldas_image

       

        

        array_wind_u.append(wind_u)

        array_wind_v.append(wind_v)

        

        array_t.append(gfs_image_t)

        array_sh.append(gfs_image_sh)

        array_rh.append(gfs_image_rh)

        array_tp.append(gfs_image_tp)



        array_rn.append(gldas_image_rn)

        

        image_reglession_u.append(np.nanmean(np.array(array_wind_u), axis=0))

        image_reglession_v.append(np.nanmean(np.array(array_wind_v), axis=0))

        image_reglession_rn.append(np.nanmean(np.array(array_rn), axis=0))

        

        image_reglession_t.append(np.nanmean(np.array(array_t), axis=0))

        image_reglession_sh.append(np.nanmean(np.array(array_sh), axis=0))

        image_reglession_tp.append(np.nanmean(np.array(array_tp), axis=0))

        image_reglession_rh.append(np.nanmean(np.array(array_rh), axis=0))





        

image_reglession_u = np.nanmean(np.array(image_reglession_u), axis=0)

image_reglession_v = np.nanmean(np.array(image_reglession_v), axis=0)

image_reglession_rn = np.nanmean(np.array(array_rn), axis=0)



image_reglession_t = np.nanmean(np.array(array_t), axis=0)

image_reglession_sh = np.nanmean(np.array(array_sh), axis=0)

image_reglession_tp = np.nanmean(np.array(array_tp), axis=0)

image_reglession_rh = np.nanmean(np.array(array_rh), axis=0)

sns.heatmap(image_reglession_u.reshape((148, 475)))
sns.heatmap(image_reglession_v.reshape((148, 475)))
sns.heatmap(image_reglession_t.reshape((148, 475)))

sns.heatmap(image_reglession_sh.reshape((148, 475)))
sns.heatmap(image_reglession_rh.reshape((148, 475)))
image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20190412T170633_20190418T190433.tif')

lon = []

lat = []

NO2 = []

wind_u = []

wind_v = []



temp = []

spechum = []

relhum = []

totper = []

rain = []





for i in range(image[:,:,0].shape[0]):

    for j in range(image[:,:,0].shape[1]):

        #print(image[:,:,0][i,j])

        NO2.append(image[:,:,0][i,j])

        lon.append(i)

        lat.append(j)

        wind_u.append(image_reglession_u.reshape((148, 475))[i,j])

        wind_v.append(image_reglession_v.reshape((148, 475))[i,j])

        temp.append(image_reglession_u.reshape((148, 475))[i,j])

        spechum.append(image_reglession_v.reshape((148, 475))[i,j])

        relhum.append(image_reglession_u.reshape((148, 475))[i,j])

        totper.append(image_reglession_v.reshape((148, 475))[i,j])

        rain.append(image_reglession_u.reshape((148, 475))[i,j])





        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)

wind_u = np.array(wind_u)

wind_v = np.array(wind_v)

temp = np.array(temp)

spechum = np.array(spechum)

relhum = np.array(relhum)

totpert = np.array(totper)

rain = np.array(rain)

        

results_wind = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v','temp','spechum' ,'relhum','totper','rain'])

results_wind = pd.DataFrame({

                    'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon),

                    'wind_u' : wind_u/(- min(wind_u)),

                    'wind_v' : wind_v/(- min(wind_v)),

                    'temp': temp/max(temp),

                    'spechum': spechum/max(spechum),

                    'relhum': relhum/max(relhum),

                    'totper': totper/max(totper) #,'rain': rain/max(rain)

                    })
results_wind["NO2"].fillna(0, inplace = True) 

results_wind["temp"].fillna(0, inplace = True) 

results_wind["spechum"].fillna(0, inplace = True) 

results_wind["relhum"].fillna(0, inplace = True) 

results_wind["totper"].fillna(0, inplace = True) 

#results_wind["rain"].fillna(0, inplace = True) 

data_scaled = scaler.fit_transform(results_wind)

pred_all_factors = KMeans(n_clusters=15).fit_predict(results_wind)

plt.figure()

sns.heatmap(pred_all_factors.reshape((148, 475)))
overlay_image_on_puerto_rico_df(power_plants_df, pred_all_factors.reshape((148, 475)), 12)
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
x= no2_weather_values_stats[['temp_2m_above_ground_mean','sp_humidity_2m_above_ground_mean',

                     'rh_2m_above_ground_mean','wind_velocity',#'date_from','tropospheric_NO2','no2_mean',

                     'tot_percip_surface_mean', 'aai_mean','tropopause_pressure_mean','cloud_fraction_mean', 'Qair_f_inst_mean',

                     'Tair_f_inst_mean','Wind_f_inst_mean']]

y = no2_weather_values_stats['no2_mean']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020)
lm = LinearRegression()

lm_fit = lm.fit(x_train, y_train)

lm_train_fit = lm_fit.predict(x_train)

y_pred_lm = lm_fit.predict(x_test)

#print("r2 score on Test:", r2_score(y_test,y_pred_lm))

lm.score(x_test, y_test)

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lm)))

print('MSE :', metrics.mean_squared_error(y_test, y_pred_lm))



coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])

coeff_df

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lm, 'difference': y_test - y_pred_lm})

df.head()

#lm.intercept_
# Import Eli5 package

import eli5

from eli5.sklearn import PermutationImportance



# Find the importance of columns for prediction

perm = PermutationImportance(lm_fit, random_state=2020).fit(x_test,y_pred_lm)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())

#Understanding how each feature influences the prediction

eli5.show_prediction(lm_fit, doc=x_test.iloc[[5]], feature_names=list(x_test.columns))
#Import SHAP package

import shap



#Create explainer for linear model

explainer = shap.LinearExplainer(lm_fit,data=x_test.values)

shap_values = explainer.shap_values(x_test)
#Understanding how each feature influences the prediction



shap.initjs()

ind = 55





shap.force_plot(

    explainer.expected_value, shap_values[ind,:], x_test.iloc[ind,:],

    feature_names=x_test.columns.tolist()

)



shap.summary_plot(shap_values,x_test)
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

ind_params = {'eta': 0.1, 

              'eval_metric': 'rmse',

              'n_estimators': 1000,

              'seed':0,

              'subsample': 0.8,

              'colsample_bytree': 0.8, 

             'objective': 'reg:linear'}

optimized_GBM = GridSearchCV(XGBRegressor(**ind_params), 

                            cv_params, cv = 5, n_jobs = -1) 

cv_res = optimized_GBM.fit(x_train,y_train)

optimized_GBM.cv_results_
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(max_depth = 10, n_estimators = 1000, random_state = 2020)

model_rf = forest.fit(x_train, y_train)

print(model_rf.score(x_train, y_train))

# Making predictions

y_pred_rf = model_rf.predict(x_test)

list(zip(x_train.columns,model_rf.feature_importances_))

model_rf.score(x_test,y_test)

print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf)))

print('MSE :', metrics.mean_squared_error(y_test, y_pred_rf))