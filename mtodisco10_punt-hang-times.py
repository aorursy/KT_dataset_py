#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools, offline
%matplotlib inline 
#Read in data
#Only reading in 1 week to save time/space
NGS_df = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv')
def get_hang_time(ngs_df, start_event='punt', *stop_events):
    punt_event = ngs_df.loc[ngs_df.Event==start_event] \
        .groupby(['Season_Year', 'GameKey','PlayID'], as_index = False)['Time'].min()
    punt_event.rename(columns = {'Time':'punt_time'}, inplace=True)
    punt_event['punt_time'] = pd.to_datetime(punt_event['punt_time'],\
                                             format='%Y-%m-%d %H:%M:%S.%f')
    
    receiving_event = ngs_df.loc[ngs_df.Event.isin(stop_events)] \
        .groupby(['Season_Year', 'GameKey','PlayID'], as_index = False)['Time'].min()
    receiving_event.rename(columns = {'Time':'receiving_time'}, inplace=True)
    receiving_event['receiving_time'] = pd.to_datetime(receiving_event['receiving_time'],\
                                             format='%Y-%m-%d %H:%M:%S.%f')
    
    punt_df = punt_event.merge(receiving_event, how='inner', on = ['Season_Year','GameKey','PlayID']) \
                .reset_index(drop=True)
    
    punt_df['hang_time'] = (punt_df['receiving_time'] - punt_df['punt_time']).dt.total_seconds()
    
    return punt_df
punt_df = get_hang_time(NGS_df, 'punt', 'punt_received', 'fair_catch')
#Show histogram of the hang_time column
punt_df.hang_time.hist();
print('The average hang time of a punt is {} seconds' .format(round(punt_df['hang_time'].mean(), 1)))
print('The median hang time of a punt is {} seconds' .format(round(punt_df['hang_time'].median(), 1)))
print(str(round(len(punt_df.loc[punt_df.hang_time < 5.5]) / len(punt_df) * 100, 1)) \
    + '% of hang times are less than 5 1/2 seconds')
