import this
import pandas as pd
import numpy as np

import os
print(os.listdir("../input"))
mast_data = pd.read_csv('../input/demo_mast.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
mast_data.head()
ref_data = pd.read_csv('../input/demo_refs.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
ref_data.head()
# what is this df object?
type(mast_data)
# get num rows and columns# get n 
mast_data.shape
# shape is an attribute, not a function
mast_data.shape()
mast_data.info()
mast_data.head()
mast_data.tail()
mast_data.describe()
# subset a single column with a column label
mast_data['SPD_59_COMB_AVG']
mast_data.SPD_59_COMB_AVG
mast_data.SPD_59_COMB_AVG.head()
# delete columns
# this won't drop in-place unless you use the inplace parameter
mast_data.drop('SPD_59_COMB_AVG', axis=1).head()
# Stamp is the index, NOT a column
mast_data.Stamp # this will fail
mast_data.index
# first row
mast_data.loc['2014-12-17 11:00:00']
# 5th row
mast_data.loc['2014-12-17 11:40:00']
# first row
mast_data.iloc[0]
# 5th row
mast_data.iloc[4]
# last row
mast_data.iloc[-1]
# the bracket notation
# row subsetter
# comma
# column subsetter
mast_data.loc['2014-12-17 11:10:00', 'SPD_59_COMB_AVG']
mast_data.loc['2014-12-17 11:00:00':'2014-12-17 12:00:00', ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]
mast_data.loc[mast_data.index < '2014-12-17 12:00:00', ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]
mast_data.loc[mast_data.index.month == 12, ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]
mast_data.index.year.unique()
mast_data.groupby(mast_data.index.year).mean()

mast_data.groupby([mast_data.index.year, mast_data.index.month]).mean()
annual_profile = mast_data.groupby(mast_data.index.month).mean().SPD_59_COMB_AVG
annual_profile
mast_data.resample('MS').mean()
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
%matplotlib inline
annual_profile.plot();
# annual_profile.plot(ylim=[0,10], title='Annual wind speed profile');
import plotly
import plotly.graph_objs as go
plotly.__version__
fig = go.FigureWidget(data=[{'x':annual_profile.index,'y':annual_profile.values}])
fig
ref_data.head()
yearly_monthly_means = ref_data.resample('MS').mean()
monthly_means = ref_data.groupby(ref_data.index.month).mean()
monthly_means_for_normal = monthly_means.loc[yearly_monthly_means.index.month]
monthly_means_for_normal.index = yearly_monthly_means.index
yearly_monthly_means_normal = yearly_monthly_means/monthly_means_for_normal
yearly_monthly_means_normal_rolling = yearly_monthly_means_normal.rolling(12, center=True, min_periods=10).mean()
yearly_monthly_means_normal_rolling.head(10)
annual_profile.plot(ylim=[0,10])
import plotly.graph_objs as go

data = [{'x':yearly_monthly_means_normal_rolling.index,'y':yearly_monthly_means_normal_rolling[ref], 'name':ref} for ref in yearly_monthly_means_normal_rolling.columns]
fig = go.FigureWidget(data=data)
fig
fig.layout.title = 'Normalized 12-month rolling average'
# remove references 9 and 10
ref_data.loc[:,'1':'8'].head(10)
# remove references 9 and 10
ref_data.drop(['9','10'], axis=1).head(10)
mast_data.head()
anemometers = ['SPD_59_COMB_AVG','SPD_47_COMB_AVG','SPD_32_COMB_AVG','SPD_15_COMB_AVG']
heights = [59,47,32,15]
ano_data = mast_data.loc[:,anemometers]
ano_data.head()
ano_data = ano_data.dropna()
ano_data.head()
ws = ano_data.mean().values
ano_data.mean()
from scipy import stats
alpha, intercept, r_value, p_value, std_err = stats.linregress(np.log(heights),np.log(ws))
alpha, intercept, r_value**2, p_value, std_err
print(f'Alpha: {alpha:.3f}; R2: {r_value**2:.4f}; Std error: {std_err*100.0:.2f}%')
# select the column you'd like to use from the mast data
# select the column you'd like to use from the reference data
site_corr_data = mast_data.SPD_59_COMB_AVG
ref_corr_data = ref_data.loc[:,'1':'8']
# resample to monthly averages
site_corr_data = site_corr_data.resample('MS').mean()
ref_corr_data = ref_corr_data.resample('MS').mean()
# concatenate into a single dataframe
corr_data = pd.concat([site_corr_data, ref_corr_data], axis=1)
corr_data = corr_data.dropna()
corr_data.head(10)
results = []
for ref in ref_corr_data.columns:
    temp_results = stats.linregress(corr_data[ref],corr_data.SPD_59_COMB_AVG)
    results.append(temp_results)
results
results = [stats.linregress(corr_data[ref],corr_data.SPD_59_COMB_AVG) for ref in ref_corr_data.columns]
results
results = pd.DataFrame.from_dict(results)
results.index = ref_corr_data.columns
results
# add a new column
results['r2'] = results.rvalue**2
results
mast_data.head()
ws = mast_data.SPD_59_COMB_AVG.dropna()
ws.head()
freq_dist = ws.groupby(ws.round()).count()
freq_dist/ws.size*100.0
bins = pd.cut(ws, bins=np.arange(0,26,0.5))
ws.groupby(bins).count()/ws.size*100.0
vane = mast_data.DIR_80_AVG.dropna()
vane.head()
vane.describe()
vane = vane.replace(360.0,0.0)
sectors = 12
bin_width = 360/sectors
dir_bins = np.floor_divide(np.mod(vane + (bin_width/2.0),360.0),bin_width)
print(f'Number of direction sectors: {sectors}; Sector bin width: {bin_width}')
dir_bins.tail(15)
wind_rose = vane.groupby(dir_bins).count()
wind_rose
dir_edges = np.append(np.append([0],np.arange(bin_width/2, 360+bin_width/2, bin_width)),360)
dir_labels = np.arange(0,361,bin_width)
# dir_edges

dir_bins = pd.cut(vane, bins=dir_edges)
dir_bins.sort_values().unique()

# dir_bins = pd.cut(vane, bins=dir_edges, right=False, labels=dir_labels) #zero inclusive
# vane.groupby(dir_bins).count()
ws.plot()
vane.plot(figsize=[20,5], style='.');
# Colors for plotting
EDFGreen = '#509E2F'
EDFOrange = '#FE5815'
EDFBlue = '#001A70'

ws.plot(figsize=[20,5], color=EDFBlue, title='Wind speed')
corr_data.head()
corr_data.plot(kind='scatter', x='1', y='SPD_59_COMB_AVG', xlim=[0,10], ylim=[0,10], color=EDFBlue, title='Monthly wind speed correlation')
freq_dist.plot(kind='bar');
fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(111)
freq_dist.plot(kind='bar', color=EDFBlue, ax=ax)
ax.set_ylabel('Bin count')
ax.set_xlabel('Wind speed bin [m/s]')
ax.set_title('Frequency distribution')
ax.set_xticks(np.arange(0,26,5))
ax.set_xticklabels(np.arange(0,26,5))
plt.show()
ref = '3'
fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111)
corr_data.plot(kind='scatter', x=ref, y='SPD_59_COMB_AVG', color=EDFBlue, ax=ax)
ax.plot([0,10], np.array([0,10])*results.loc[ref,'slope']+results.loc[ref,'intercept'], color=EDFGreen)
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_xlabel('Reference [m/s]')
ax.set_ylabel('Site [m/s]')
ax.set_title(f"Monthly wind speed correlation (R2: {results.loc[ref,'r2']:.3f})")
plt.show()
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')
dir_bin_width_radians = np.radians(bin_width)
ax.set_theta_direction('clockwise')
ax.set_theta_zero_location('N')
ax.bar(np.radians(wind_rose.index.values*bin_width), wind_rose.values, width=dir_bin_width_radians, color=EDFGreen)
ax.set_title('Wind rose')
ax.set_yticklabels([])
ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
plt.show()
import plotly
print(plotly.__version__)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)
fig = go.FigureWidget(data=[{'x':freq_dist.index, 'y':freq_dist.values, 'type':'bar'}])
fig
fig.layout.title = 'Wind speed frequency distribution'
fig.layout.margin.t = 25
fig.layout.width = 900
fig.layout.height = 500
fig.layout.xaxis.title = 'Wind speed [m/s]'
fig.layout.yaxis.title = 'Frequency [count]'
fig.layout.yaxis.tickvals = []
fig = go.FigureWidget(
    data=[{'x':corr_data[ref], 
           'y':corr_data.SPD_59_COMB_AVG, 
           'type':'scatter', 
           'mode':'markers', 
           'name':'Wind speeds',
           'marker':{'color':EDFBlue,
                     'size':8},
           'text':corr_data.index.strftime('%Y-%m'),
           'hoverinfo':'text'},
          {'x':[0,10], 
           'y':np.array([0,10])*results.loc[ref,'slope']+results.loc[ref,'intercept'], 
           'type':'scatter', 
           'mode':'lines', 
           'name':'Fit',
           'line':{'color':EDFGreen,
                   'width':5}}
         ],
    layout={'title':'Monthly wind speed correlation',
            'width':600,
            'font':{'size':14},
            'margin':{'t':30,'b':35,'r':0,'l':35},
            'xaxis':{'rangemode':'tozero',
                     'title':'Reference [m/s]'},
            'yaxis':{'rangemode':'tozero',
                     'title':'Site [m/s]'}})
fig
mast_data.to_csv('mast_output.csv')
mast_data.to_parquet('mast_output.parquet')
print(os.listdir("../working"))
%%timeit
pd.read_csv('mast_output.csv')
%%timeit
pd.read_parquet('mast_output.parquet')
import anemoi as an
an.__version__
mast_data.columns.name = 'sensor'
mast_data.head()
mast = an.MetMast(data=mast_data, name='Example mast', primary_ano='SPD_59_COMB_AVG', primary_vane='DIR_80_AVG')
mast.data.head()
mast.metadata
# an.analysis.
shear = an.analysis.shear.mast_annual(mast)
shear
# ?an.analysis.correlate.ws_correlation_orthoginal_distance_model()
# ??an.analysis.correlate.ws_correlation_orthoginal_distance_model()
corr_data.head()
an.analysis.correlate.ws_correlation_orthoginal_distance_model(corr_data, ref_ws_col='1', site_ws_col='SPD_59_COMB_AVG', force_through_origin=False)
# an.plotting.
shear_fig = an.plotting.shear.annual_mast_results(shear)
offline.iplot(shear_fig)
