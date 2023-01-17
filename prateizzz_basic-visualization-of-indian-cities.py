# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
data.shape
data_India  = data[ (data['Country'] == 'India') & (data['Year'] < 2020)]
India_plot=data_India.groupby(['City','Year'])['AvgTemperature'].mean().reset_index()
India_plot.pivot('Year','City','AvgTemperature').plot()
plt.gcf().set_size_inches(14,6)
def season(df):
    if df in [12,1,2] :
        return 'Winter'
    elif  df in [3,4,5]:
        return 'Summer'
    elif df in [6,7,8]:
        return 'Monsoon'
    elif df in [9,10,11]:
        return 'Autumn'
    else:
        return 'NA'
pd.options.mode.chained_assignment = None  # default='warn'
data_India['Season'] = data_India['Month'].apply(season)
data_India['AvgTemperature']=data_India['AvgTemperature'].astype('float64')
data_India[['Month' , 'Day' , 'Year']]=data_India[['Month' , 'Day' , 'Year']].astype('int64')


data_Mumbai = data_India[data_India['City'] == 'Bombay (Mumbai)']
Mumbai_plot=data_Mumbai.groupby(['Season','Year'])['AvgTemperature'].mean().reset_index()
Mumbai_plot.pivot('Year','Season','AvgTemperature').plot()
plt.gcf().set_size_inches(14,6)
data_Delhi = data_India[data_India['City'] == 'Delhi']
Delhi_plot=data_Delhi.groupby(['Season','Year'])['AvgTemperature'].mean().reset_index()
Delhi_plot.pivot('Year','Season','AvgTemperature').plot()
plt.gcf().set_size_inches(14,6)
data_Calcutta    = data_India[data_India['City'] == 'Calcutta']
Calcutta_plot=data_Calcutta.groupby(['Season','Year'])['AvgTemperature'].mean().reset_index()
Calcutta_plot.pivot('Year','Season','AvgTemperature').plot()
plt.gcf().set_size_inches(14,6)
data_Chennai    = data_India[data_India['City'] == 'Chennai (Madras)']
Chennai_plot=data_Chennai.groupby(['Season','Year'])['AvgTemperature'].mean().reset_index()
Chennai_plot.pivot('Year','Season','AvgTemperature').plot()
plt.gcf().set_size_inches(14,6)
data_Winter = data_India[data_India['Season'] == 'Winter']
Winter_plot=data_Winter.groupby(['City','Year'])['AvgTemperature'].mean().reset_index()
Winter_plot.pivot('Year','City','AvgTemperature').plot()
plt.title('Cities in Winter Season')
plt.gcf().set_size_inches(14,6)
data_Summer = data_India[data_India['Season'] == 'Summer']
Summer_plot=data_Summer.groupby(['City','Year'])['AvgTemperature'].mean().reset_index()
Summer_plot.pivot('Year','City','AvgTemperature').plot()
plt.title('Cities in Summer Season')
plt.gcf().set_size_inches(14,6)
data_Monsoon = data_India[data_India['Season'] == 'Monsoon']
Monsoon_plot=data_Monsoon.groupby(['City','Year'])['AvgTemperature'].mean().reset_index()
Monsoon_plot.pivot('Year','City','AvgTemperature').plot()
plt.title('Cities in Monsoon Season')
plt.gcf().set_size_inches(14,6)
data_Autumn = data_India[data_India['Season'] == 'Autumn']
Autumn_plot=data_Autumn.groupby(['City','Year'])['AvgTemperature'].mean().reset_index()
Autumn_plot.pivot('Year','City','AvgTemperature').plot()
plt.title('Cities in Autumn Season')
plt.gcf().set_size_inches(14,6)


