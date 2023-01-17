# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import matplotlib
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn as sns; 
from matplotlib.ticker import FormatStrFormatter
%matplotlib inline
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset('noaa_icoads', project='bigquery-public-data')
dset = client.get_dataset(dataset_ref)
[i.table_id for i in client.list_tables(dset)]
icoads_core_2017 = client.get_table(dset.table('icoads_core_2017'))
[i.name+", type: "+i.field_type for i in icoads_core_2017.schema]
# longitude 92-100, latitude 5-14 
QUERY = """
        SELECT latitude, longitude, timestamp
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2017`
        WHERE longitude >= 92 AND longitude <= 100 AND latitude >= 5 AND latitude <= 14
        """
df = client.query(QUERY).to_dataframe()
print(df.latitude.size, df.longitude.size)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

map = Basemap(projection='ortho',lat_0=12,lon_0=96,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='coral',lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
# make up some data on a regular lat/lon grid.
lats = df['latitude'].values
lons = df['longitude'].values
x, y = map(lons, lats)
# contour data over the map.
cs = map.scatter(x,y)
plt.title('contour lines over filled continent background')
plt.show()
longmin=90
longmax=100
lamin=4
lamax=14
longmin
#QUERY = """
#       SELECT  timestamp, sea_surface_temp, wind_speed, visibility, present_weather, sea_level_pressure, air_temperature, wetbulb_temperature, dewpoint_temperature
#      FROM `bigquery-public-data.noaa_icoads.icoads_core_1662_2000`
#     WHERE longitude >= 92 AND longitude <= 100 AND latitude >= 5 AND latitude <= 14
#    """
#df1=client.query(QUERY).to_dataframe()
#print(df1.latitude.size, df1.longitude.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2001_2004`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df2=client.query(QUERY).to_dataframe()
print(df2.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2005`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df3=client.query(QUERY).to_dataframe()
print(df3.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2006`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df4=client.query(QUERY).to_dataframe()
print(df4.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2007`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df5=client.query(QUERY).to_dataframe()
print(df5.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2008`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df6=client.query(QUERY).to_dataframe()
print(df6.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2009`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df7=client.query(QUERY).to_dataframe()
print(df7.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2010`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df8=client.query(QUERY).to_dataframe()
print(df8.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2011`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df9=client.query(QUERY).to_dataframe()
print(df9.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2012`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df10=client.query(QUERY).to_dataframe()
print(df10.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2013`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df11=client.query(QUERY).to_dataframe()
print(df11.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2014`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df12=client.query(QUERY).to_dataframe()
print(df12.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2015`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df13=client.query(QUERY).to_dataframe()
print(df13.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2016`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df14=client.query(QUERY).to_dataframe()
print(df14.size)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.noaa_icoads.icoads_core_2017`
        WHERE longitude >= 90 AND longitude <= 100 AND latitude >= 4 AND latitude <= 14
        """
df15=client.query(QUERY).to_dataframe()
print(df15.size)

#dfl=df1.append(df2, ignore_index = True) 


dfl=df2.append(df3, ignore_index = True) 
dfl=dfl.append(df4, ignore_index = True) 
dfl=dfl.append(df5, ignore_index = True) 
dfl=dfl.append(df6, ignore_index = True) 
dfl=dfl.append(df7, ignore_index = True) 
dfl=dfl.append(df8, ignore_index = True) 
dfl=dfl.append(df9, ignore_index = True) 
dfl=dfl.append(df10, ignore_index = True) 
dfl=dfl.append(df11, ignore_index = True) 
dfl=dfl.append(df12, ignore_index = True) 
dfl=dfl.append(df13, ignore_index = True) 
dfl=dfl.append(df14, ignore_index = True) 
dfl=dfl.append(df15, ignore_index = True) 

dft=dfl.copy()
dft

dfl.size
dfl
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'imma_version',
                                 'attm_count',
                                 'time_indicator',
                                 'latlong_indicator',
                                 'ship_course',
                                 'ship_speed',
                                 'national_source_indicator',
                                 'id_indicator',
                                 'callsign'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
dft2
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 
                                 'attm_count',
                                 
                                 
                                 'ship_course',
                                 'ship_speed',
                                 
                                 'id_indicator'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
corr = dft2.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'country_code',
                                 'wind_direction_indicator',
                                 'wind_direction_true',
                                 'wind_speed_indicator',
                                 'wind_speed',
                                 'visibility_indicator',
                                 'visibility',
                                 'present_weather',
                                 'past_weather',
                                 'sea_level_pressure',
                                 'characteristic_of_ppp',
                                 'amt_pressure_tend',
                                 'indicator_for_temp',
                                 'air_temperature',
                                 'wbt_indicator',
                                 'wetbulb_temperature',
                                 'dpt_indicator'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
dft2
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'country_code',
                                 'wind_direction_indicator',
                                 'wind_direction_true',
                                 'wind_speed_indicator',
                                 'wind_speed',
                                 'visibility_indicator',
                                 'visibility',
                                 'present_weather',
                                 'past_weather',
                                 'sea_level_pressure',
                                 'characteristic_of_ppp',
                                 'amt_pressure_tend',
                                 'indicator_for_temp',
                                 'air_temperature',
                                 'wbt_indicator',
                                 'wetbulb_temperature',
                                 'dpt_indicator'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
corr = dft2.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'dewpoint_temperature',
                                 'sst_measurement_method',
                                 'total_cloud_amount',
                                 'lower_cloud_amount',
                                 'low_cloud_type',
                                 'cloud_height',
                                 'cloud_height',
                                 'middle_cloud_type',
                                 'high_cloud_type',
                                 'wave_direction',
                                 'wave_period',
                                 'wave_height',
                                 'swell_direction',
                                 'swell_period',
                                 'swell_height',
                                 'box_system_indicator',
                                 'ten_degree_box_number'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
dft2
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'dewpoint_temperature',
                                 'sst_measurement_method',
                                 'total_cloud_amount',
                                 'lower_cloud_amount',
                                 'low_cloud_type',
                                 'cloud_height',
                                 'cloud_height',
                                 'middle_cloud_type',
                                 'high_cloud_type',
                                 'wave_direction',
                                 'wave_period',
                                 'wave_height',
                                 'swell_direction',
                                 'swell_period',
                                 'swell_height',
                                 'box_system_indicator',
                                 'ten_degree_box_number'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
corr = dft2.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'one_degree_box_number',
                                 'deck',
                                 'source_id',
                                 'platform_type',
                                 'dup_status',
                                 'dup_check',
                                 'track_check',
                                 'pressure_bias',
                                 'wave_period_indicator',
                                 'swell_period_indicator',
                                 'second_country_code',
                                 'adaptive_qc_flags',
                                 'nightday_flag',
                                 'trimming_flags',
                                 'ncdc_qc_flags'
                                 
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)

dft2
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'one_degree_box_number',
                                 'deck',
                                 'source_id',
                                 'platform_type',
                                 'dup_status',
                                 'dup_check',
                                 
                                 'trimming_flags',
                                 'ncdc_qc_flags'
                                 
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)
dft2=dft2.dropna()
corr = dft2.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 'external',
                                 'landlocked_flag',
                                 'source_exclusion_flags',
                                 'unique_report_id',
                                 'release_no_primary',
                                 'release_no_secondary',
                                 'release_no_tertiary',
                                 'release_status_indicator',
                                 'intermediate_reject_flag'
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)

dft2
dft2=dft.copy()
dft2.drop(dft2.columns.difference(['sea_surface_temp',
                                 
                                
                               
                                 'unique_report_id',
                                 'release_no_primary',
                                 'release_no_secondary',
                                 'release_no_tertiary',
                                 'release_status_indicator',
                                 
                                 ]), 1, inplace=True)
cols = dft2.columns.tolist()
cols.insert(0, cols.pop(cols.index('sea_surface_temp')))
dft2 = dft2.reindex(columns= cols)

dft2=dft2.dropna()
corr = dft2.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
dft.drop(dft.columns.difference(['timestamp',
                               'sea_surface_temp',
                               'swell_direction',
                                 'swell_period',
                                 'swell_height',
                                 'release_no_tertiary'
                                ]), 1, inplace=True)
dft = dft.set_index('timestamp')
dft.index
dft=dft.dropna()
dft
dft1=dft.drop(columns=[ 'swell_direction',
                                 'swell_period',
                                 'swell_height',
                                 'release_no_tertiary'])
dft1
dft2=dft.drop(columns=[ 'sea_surface_temp'])
dft2
dft1 = dft1['sea_surface_temp'].resample('MS').mean()
dft2 = dft2.resample('MS').mean()
dft2=dft2.reset_index()
del dft2['timestamp']
dft2
dft1.plot(figsize=(15, 6))
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(dft1, model='additive')
fig = decomposition.plot()
plt.show()
decomposition.resid.mean()
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        #try:
            mod = sm.tsa.statespace.SARIMAX(endog=dft1,
                                            exog=dft2.values,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
       # except:
        #        continue
mod = sm.tsa.statespace.SARIMAX(endog=dft1.values,
                                    exog=dft2.values,
                                order=(1, 0, 0),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

results.plot_diagnostics(figsize=(16, 8))
plt.show()
predict = results.get_prediction(start=100, dynamic=False) # , end = pd.to_datetime('2017-01-01')
predict_ci = predict.conf_int()
dataset = pd.DataFrame({'Column1': predict_ci[:, 0], 'Column2': predict_ci[:, 1]})
temp=dft1.iloc[100:]
temp=temp.to_frame()
temp=temp.reset_index()
temp=temp.join(dataset)
del temp['sea_surface_temp']
temp = temp.set_index('timestamp')

temp['mean'] = temp.mean(axis=1)

predict_ci=temp
temp
# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='temperature', xlabel='Date', ylabel='temperature')

# Plot data points
ax = dft1.plot(label='observed')

# Plot predictions
#predict.predicted_mean.loc['2015-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['2015-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_ci.iloc[:,2].loc['2015-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (2015)')
ci = predict_ci.loc['2015-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')
df_forecasted = predict_ci.iloc[:,2]
df_truth = dft1['2015-01-01':]
mse = ((df_forecasted - df_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))



dfl.drop(dfl.columns.difference(['timestamp',
                               'sea_surface_temp',
                               'wind_speed',
                               'visibility',
                               'present_weather',
                               'sea_level_pressure',
                               'air_temperature',
                               'wetbulb_temperature',
                               'dewpoint_temperature'
                                ]), 1, inplace=True)
dfl = dfl.set_index('timestamp')
dfl.index
dfl

dfl=dfl.dropna()
dfl
dfl1=dfl
dfl1=dfl1.drop(columns=[ 'wind_speed', 'visibility', 'present_weather', 'sea_level_pressure', 'air_temperature', 'wetbulb_temperature', 'dewpoint_temperature'])
dfl1
dfl2=dfl
dfl2=dfl2.drop(columns=['sea_surface_temp'])
dfl1 = dfl1['sea_surface_temp'].resample('MS').mean()
#show air temperature
#dfbobo=dfl
#dfbobo=dfbobo.dropna()
#dfbobo.drop(dfbobo.columns.difference(['air_temperature']), 1, inplace=True)
#dfbobo
#dfbobo = dfbobo['air_temperature'].resample('MS').mean()
#dfbobo.plot(figsize=(15, 6))
dfl1.plot(figsize=(15, 6))
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(dfl1, model='additive')
fig = decomposition.plot()
plt.show()

dfl
dfl3=dfl
dfl3 = dfl3.resample('MS').mean()
dfl3=dfl3.reset_index()
dfl3
del dfl3['timestamp']
del dfl3['sea_surface_temp']
dfl3
dfl3.values.size
for param in pdq:
    for param_seasonal in seasonal_pdq:
        #try:
            mod = sm.tsa.statespace.SARIMAX(endog=dfl1,
                                            exog=dfl3.values,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
       # except:
        #        continue
mod = sm.tsa.statespace.SARIMAX(endog=dfl1.values,
                                    exog=dfl3.values,
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

results.plot_diagnostics(figsize=(16, 8))
plt.show()



pd.to_datetime('2015-01-01')
dfl1
predict = results.get_prediction(start=100, dynamic=False) # , end = pd.to_datetime('2017-01-01')
predict_ci = predict.conf_int()

dataset = pd.DataFrame({'Column1': predict_ci[:, 0], 'Column2': predict_ci[:, 1]})
temp=dfl1.iloc[100:]
temp=temp.to_frame()
temp=temp.reset_index()
temp=temp.join(dataset)
del temp['sea_surface_temp']
temp = temp.set_index('timestamp')

temp['mean'] = temp.mean(axis=1)

predict_ci=temp
temp
# Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='temperature', xlabel='Date', ylabel='temperature')

# Plot data points
ax = dfl1.plot(label='observed')

# Plot predictions
#predict.predicted_mean.loc['2015-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['2015-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_ci.iloc[:,2].loc['2015-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (2015)')
ci = predict_ci.loc['2015-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')
df_forecasted = predict_ci.iloc[:,2]
df_truth = dfl1['2015-01-01':]
mse = ((df_forecasted - df_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
dfl1
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(dfl1, model='additive')
fig = decomposition.plot()
plt.show()

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(dfl1,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
                continue
mod = sm.tsa.statespace.SARIMAX(dfl1,
                                    
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()


results.plot_diagnostics(figsize=(16, 8))
plt.show()


pred = results.get_prediction(start=150, dynamic=False) # , end = pd.to_datetime('2017-01-01')
pred_ci = pred.conf_int()
ax = dfl1.plot(label='observed')
#pred=pd.Series(pred)
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('timestamp')
ax.set_ylabel('sea_surface_temp')
plt.legend()
plt.show()
df_forecasted = pred.predicted_mean
df_truth = dfl1['2015-01-01':]
mse = ((df_forecasted - df_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=600)
pred_ci = pred_uc.conf_int()
ax = dfl1.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('temperature')
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend()
plt.show()
pred_ci
pred_ci['mean'] = pred_ci.mean(axis=1)
pred_ci
pred_ci.drop(pred_ci.columns.difference(['mean']), 1, inplace=True)
pred_ci
pred_ci=pred_ci-28
pred_ci
pred_ci.plot()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(pred_ci)
pred_ci=pred_ci.reset_index()
plot_ci=pred_ci.loc[0:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[0:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci
plot_ci=pred_ci.loc[0:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[1:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[2:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[3:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[4:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[5:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[6:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[7:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[8:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[9:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[10:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
plot_ci=pred_ci.loc[11:600:12]  
plot_ci=plot_ci.set_index('index')
plot_ci.plot()
