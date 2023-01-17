import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
# remove obvious outliers

allLat = np.array(list(df['pickup_latitude']) + 

                     list(df['dropoff_latitude']))

allLong = np.array(list(df['pickup_longitude']) +

                      list(df['dropoff_longitude']))



longLimits = [np.percentile(allLong, 0.3), 

              np.percentile(allLong, 99.7)]

latLimits = [np.percentile(allLat, 0.3),

             np.percentile(allLat, 99.7)]

durLimits = [np.percentile(df['trip_duration'], 0.4),

             np.percentile(df['trip_duration'], 99.7)]
df = df[(df['pickup_latitude']   >= latLimits[0] ) & (df['pickup_latitude']   <= latLimits[1]) ]

df = df[(df['dropoff_latitude']  >= latLimits[0] ) & (df['dropoff_latitude']  <= latLimits[1]) ]

df = df[(df['pickup_longitude']  >= longLimits[0]) & (df['pickup_longitude']  <= longLimits[1])]

df = df[(df['dropoff_longitude'] >= longLimits[0]) & (df['dropoff_longitude'] <= longLimits[1])]

df = df[(df['trip_duration']     >= durLimits[0] ) & (df['trip_duration']     <= durLimits[1]) ]

df = df.reset_index(drop=True)
def rounding(l):

    return float('{0:.3f}'.format(l))
df['p_lat'] = df['pickup_latitude'].map(rounding)

df['p_long'] = df['pickup_longitude'].map(rounding)

df['d_lat'] = df['dropoff_latitude'].map(rounding)

df['d_long'] = df['dropoff_longitude'].map(rounding)
df['pickup_datetime'] = df['pickup_datetime'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

df['dropoff_datetime'] = df['dropoff_datetime'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
df['hour'] = df['pickup_datetime'].map(lambda x: x.hour)

df['dayofweek'] = df['pickup_datetime'].map(lambda x: x.dayofweek)
def plot_net_flow(data_new):

    df_p = data_new[['p_lat', 'p_long']]

    df_d = data_new[['d_lat', 'd_long']]

    df_p_pivoted = df_p.pivot_table(index='p_lat', columns='p_long', aggfunc=len, fill_value=0)

    df_d_pivoted = df_d.pivot_table(index='d_lat', columns='d_long', aggfunc=len, fill_value=0)

    

    df_p_pivoted.columns.name=None

    df_p_pivoted.index.name=None

    df_d_pivoted.columns.name=None

    df_d_pivoted.index.name=None

    

    df_sub = df_p_pivoted.subtract(df_d_pivoted,fill_value=0)

    

    df_sub_unpivot = df_sub.unstack().reset_index(name='value')

    

    df_sub_unpivot = df_sub_unpivot.rename(columns={'level_0':'long', 'level_1':'lat', 'value':'net flow'})

    

    df_sub_unpivot_pos = df_sub_unpivot[df_sub_unpivot['net flow']>0]

    df_sub_unpivot_neg = df_sub_unpivot[df_sub_unpivot['net flow']<0]

    

    plt.figure(figsize = (10,10))

    plt.plot(df_sub_unpivot['long'],df_sub_unpivot['lat'],'.',alpha = 0.3,markersize=1, color='y')

    plt.plot(df_sub_unpivot_pos['long'], df_sub_unpivot_pos['lat'], 'o', markersize=2,color='b')

    plt.plot(df_sub_unpivot_neg['long'], df_sub_unpivot_neg['lat'], 'o',markersize=2,color='r')
df_new = df[(df['hour'] > 16) & (df['dayofweek'] < 5)]

plot_net_flow(df_new)
df_new = df[(df['hour'] < 10) & (df['dayofweek'] < 5)]

plot_net_flow(df_new)