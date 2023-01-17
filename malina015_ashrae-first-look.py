import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from bokeh import palettes as bh

from datetime import datetime
folder = '../input/ashrae-energy-prediction/'

weather_train_df = pd.read_csv(folder + 'weather_train.csv')

building_meta_df = pd.read_csv(folder + 'building_metadata.csv')

train_df = pd.read_csv(folder + 'train.csv')

## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

## REducing memory



weather_train_df = reduce_mem_usage(weather_train_df)

building_meta_df = reduce_mem_usage(building_meta_df)

train_df = reduce_mem_usage(train_df)

merged_train = pd.merge(train_df, building_meta_df, how="left", on=["building_id"])

print('The shape of our data is:', building_meta_df.shape)

print(f"There are {len(building_meta_df['building_id'].unique())} distinct buildings." )
building_meta_df.describe()
building_meta_pu = building_meta_df.groupby('primary_use').agg({'building_id':'count',

                                             'year_built':['min','max','mean'],

                                            'square_feet':['min','max','mean']})

building_meta_pu_mi = building_meta_pu.columns

building_meta_pu_mi = pd.Index([e[0] + ' ' + e[1] for e in building_meta_pu_mi.tolist()])

building_meta_pu.columns = building_meta_pu_mi

building_meta_pu = building_meta_pu.sort_values('building_id count', ascending = False)

building_meta_pu
fig, ax = plt.subplots()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.hist(building_meta_df['square_feet'], 20, facecolor=bh.magma(6)[1], alpha=0.75)

plt.xlabel('Size in square feet')

plt.ylabel('Number of buildings')

plt.title('Histogram of size in square feet')

plt.show()
fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))



ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['square_feet'],

                     20, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])

ax1.legend(prop={'size': 10})

a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['square_feet'],

                     20, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])

ax2.legend(prop={'size': 10})

a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['square_feet'],

                     20, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])

ax3.legend(prop={'size': 10})

a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['square_feet'],

                     20, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])

ax4.legend(prop={'size': 10})

ax1.set_xlim([0, 500000])

plt.show()
fig, ax = plt.subplots()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.hist(building_meta_df['year_built'], 20, facecolor=bh.magma(6)[2], alpha=0.75)

plt.xlabel('Year of building')

plt.ylabel('Number of buildings')

plt.title('Histogram of year of building')

plt.show()
fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))



ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['year_built'],

                     20, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])

ax1.legend(prop={'size': 10})

a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['year_built'],

                     20, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])

ax2.legend(prop={'size': 10})

a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['year_built'],

                     20, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])

ax3.legend(prop={'size': 10})

a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['year_built'],

                     20, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])

ax4.legend(prop={'size': 10})

ax1.set_xlim([1900, 2020])

plt.show()
fig, ax = plt.subplots()



ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.hist(building_meta_df['floor_count'], 20, facecolor=bh.magma(6)[4], alpha=0.75)

plt.xlabel('Number of floor')

plt.ylabel('Number of buildings')

plt.title('Histogram of floor count')

plt.show()
fig, (ax1,ax2,ax3,ax4)  = plt.subplots(4, sharex=True, figsize=(12,10))



ax1.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[0]]['floor_count'],

                     10, facecolor=bh.viridis(6)[1], alpha=0.75, label = building_meta_pu.index[0])

ax1.legend(prop={'size': 10})

a2 = ax2.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[1]]['floor_count'],

                     10, facecolor=bh.viridis(6)[2], alpha=0.75, label = building_meta_pu.index[1])

ax2.legend(prop={'size': 10})

a3 = ax3.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[2]]['floor_count'],

                     10, facecolor=bh.viridis(6)[3], alpha=0.75, label = building_meta_pu.index[2])

ax3.legend(prop={'size': 10})

a4 = ax4.hist(building_meta_df[building_meta_df['primary_use']==building_meta_pu.index[3]]['floor_count'],

                     10, facecolor=bh.viridis(6)[4], alpha=0.75, label = building_meta_pu.index[3])

ax4.legend(prop={'size': 10})

ax1.set_xlim([0, 25])

plt.show()
print('The shape of our data is:', weather_train_df.shape)

print(f"There are {len(weather_train_df.site_id.unique())} unique site_id." )

weather_train_df.describe()
site_ids = weather_train_df['site_id'].unique()

fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,12))



for i in range(0,16):

    axs[i//4,i%4 ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['air_temperature'],

                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])

    axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))

plt.show()
site_ids = weather_train_df['site_id'].unique()

fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,10))



for i in range(0,16):

    axs[i//4,i%4 ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['wind_speed'],

                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])

    axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))

plt.show()
site_ids = weather_train_df['site_id'].unique()

fig, axs   = plt.subplots(5, sharex=True,sharey=True, figsize=(12,10))



for i in range(0,5):

    axs[i ].hist(weather_train_df[weather_train_df['site_id']==site_ids[i]]['sea_level_pressure'],

                     20, facecolor=bh.magma(19)[i], alpha=0.75, label = site_ids[i])

    axs[i].set_title(''.join(['Site id:',str(site_ids[i])]))

plt.show()
print('The shape of our data is:', train_df.shape)

print(f"There are {len(train_df.building_id.unique())} unique building_id." )

train_df.head()
train_df['meter'].unique()
train_df.groupby('meter').agg({'building_id':'nunique', 'meter_reading':sum, 'timestamp': ['min','max']})
train_df['log_meter_reading']=np.log1p(train_df['meter_reading'])
meter_ids = train_df['meter'].unique()

meter_ids_map = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3:'hotwater'}



fig, axs   = plt.subplots(2,2, sharex=True,sharey=True, figsize=(12,12))



for i in range(0,4):

    axs[i//2,i%2 ].hist(train_df[train_df['meter']==meter_ids[i]]['log_meter_reading'],

                     20, facecolor=bh.cividis(4)[i], alpha=0.75, label = meter_ids_map[meter_ids[i]])

    axs[i//2,i%2].set_title(''.join(['Logaritm of meter id:',meter_ids_map[meter_ids[i]]]))

plt.show()
len(train_df[train_df['meter_reading']==0]['meter_reading'])


merged_train.head()
merged_train['Weekday'] = merged_train['timestamp'].apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))
merged_train['Weekday_num'] = merged_train['Weekday'].apply(lambda row: row.strftime('%w'))

merged_train['Weekday'] = merged_train['Weekday'].apply(lambda row: row.strftime('%A'))

merged_train['log_meter_reading']=np.log1p(merged_train['meter_reading'])
merged_train.head()
groupby_day = merged_train.groupby(['primary_use','Weekday','Weekday_num','site_id']).agg({'log_meter_reading':['sum','mean']})

merged_train
groupby_day_mi = groupby_day.columns

groupby_day_mi = pd.Index([e[0] + ' ' + e[1] for e in groupby_day_mi.tolist()])

groupby_day.columns = groupby_day_mi

groupby_day = groupby_day.reset_index()



site_ids = list(groupby_day['site_id'].unique())

primary_uses = list(building_meta_pu[building_meta_pu['building_id count']>20].index)

fig, axs   = plt.subplots(4,4, sharex=True,sharey=True, figsize=(12,12))

width = 1/9

for i in range(0,16):

    df_site = groupby_day[groupby_day['site_id']==site_ids[i]]

    for j in range(0,8):

        axs[i//4,i%4 ].bar(df_site[df_site['primary_use']==primary_uses[j]]['Weekday_num'].astype(float)+width*j,df_site[df_site['primary_use']==primary_uses[j]]['log_meter_reading mean']

                     , facecolor=bh.viridis(8)[j], alpha=0.75, label = primary_uses[j],width=width )

        axs[i//4,i%4].set_title(''.join(['Site id:',str(site_ids[i])]))

plt.show()