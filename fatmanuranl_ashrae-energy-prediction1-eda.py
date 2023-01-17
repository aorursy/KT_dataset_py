# Set your own project id here
PROJECT_ID = 'ASHRAE - Energy Prediction1'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
!pip install seaborn==0.11.0
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import gc
print(f'''Pandas version: {pd.__version__}
NumPy version: {np.__version__}
Matplotlib version: {mpl.__version__}
Seaborn version: {sns.__version__}''')
gc.enable()
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
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
train = reduce_mem_usage(train)
weather_train = reduce_mem_usage(weather_train)
building = reduce_mem_usage(building)
train.head()
train.info()
train['timestamp'] = pd.to_datetime(train['timestamp'])
print(f'''{train['timestamp'].dtype}
{train['timestamp'].min()}
{train['timestamp'].max()}''')
train.isnull().sum()
train['meter_reading'].describe()
print('Percent of zero read values:', "%.2f"%(train[train['meter_reading']== 0].shape[0] /train.shape[0]))
train[train['meter_reading'] == 0]
print(f'''Total number of buildings: {train['building_id'].nunique()}
Number of buildings with zero readings: {train[train['meter_reading']== 0]["building_id"].value_counts().shape[0]}
Percentage: {round(train[train['meter_reading']== 0]["building_id"].value_counts().shape[0] / train['building_id'].nunique(),2)}

Number of zero readings per building: 
{train[train['meter_reading']== 0]["building_id"].value_counts().sort_values(ascending = False)}

Number of buildings with only one zero readings: {sum(train[train['meter_reading']== 0]['building_id'].value_counts() == 1)}
Percentage of buildings with only one zero reading: {round( sum(train[train['meter_reading']== 0]['building_id'].value_counts() == 1) / train[train['meter_reading']== 0]["building_id"].value_counts().shape[0] , 2)}
''')
(train[train['meter_reading']== 0]).groupby('building_id')['meter'].value_counts().sort_values(ascending = False)
print('Number of zero readings for each meter:\n',train[train['meter_reading']== 0]["meter"].value_counts(),'\n')

for i in range(train["meter"].nunique()):
    percent = round(train[train['meter_reading']== 0]["meter"].value_counts()[i] /train["meter"].value_counts()[i],2)
    print(f'% of zero reads for Meter {i}: {percent}')
train['meter'] = pd.Categorical(train['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
g = sns.FacetGrid(train[train['meter_reading']== 0], col="meter",hue = 'meter',palette='coolwarm',col_wrap=2,height=3, aspect=2)
g.map(sns.histplot, 'timestamp', bins=12)
fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(y='meter', x='meter_reading', data=train, showfliers=True)
fig, axes = plt.subplots(1, 1, figsize=(14, 6))
sns.boxplot(y='meter', x='meter_reading', data=train[train['meter_reading'] !=  0 ], showfliers=False);
sns.lineplot(data=train.groupby(['timestamp']).sum(), x="timestamp", y="meter_reading")
sns.relplot(data=train.groupby(['timestamp','meter']).sum(), x="timestamp", y="meter_reading",col="meter", hue="meter",kind="line")
sns.relplot(data=(train[train['meter'].isin(['electricity','chilledwater','hotwater'])].groupby(['timestamp','meter']).sum()), x="timestamp", y="meter_reading",col="meter", hue="meter",kind="line")
gc.collect()
building.head()
building.info()
building.describe()
building.describe(include = 'O')
print(f'Null value counts:\n{building.isnull().sum()}\n')
for col in list(building.columns):
    if building[col].isnull().sum() > 0:
        print(f'% of null in column {col}: {round(building[col].isnull().sum() / building.shape[0], 2 )}' )
building.fillna('Missing', inplace=True)
print('For Primary Usage Areas\n')
print('% of Null year_built Values Less Than 50%')
for usage in list(building['primary_use'].unique()):
    percent  = round(sum(building[building['primary_use']== usage]['year_built']== 'Missing') / building[building['primary_use']== usage].shape[0], 2)
    if percent < 0.5:
        print(f'{usage}: {percent}')
        
print('\n% of Null year_built Values Higher Than 50%')
for usage in list(building['primary_use'].unique()):
    percent  = round(sum(building[building['primary_use']== usage]['year_built']== 'Missing') / building[building['primary_use']== usage].shape[0], 2)
    if percent > 0.5:
        print(f'{usage}: {percent}')
    
print('For Primary Usage Areas\n')
print('\n% of Null floor_count Values Less Than 50%\n')
for usage in list(building['primary_use'].unique()):
    percent  = round(sum(building[building['primary_use']== usage]['floor_count']== 'Missing') / building[building['primary_use']== usage].shape[0], 2)
    if percent < 0.5:
        print(f'{usage}: {percent}')
        
print('\n% of Null floor_count Values Higher Than 50%\n')
for usage in list(building['primary_use'].unique()):
    percent  = round(sum(building[building['primary_use']== usage]['floor_count']== 'Missing') / building[building['primary_use']== usage].shape[0], 2)
    if percent > 0.5:
        print(f'{usage}: {percent}')
building.drop(['floor_count','year_built'],axis=1,inplace=True)
gc.collect()
weather_train.head()
weather_train.info()
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
print(f'''{weather_train['timestamp'].dtype}
{weather_train['timestamp'].min()}
{weather_train['timestamp'].max()}''')
print(f'Null value counts:\n{weather_train.isnull().sum()}\n')
for col in list(weather_train.columns):
    if weather_train[col].isnull().sum() > 0:
        print(f'% of null in column {col}: {round(weather_train[col].isnull().sum() / weather_train.shape[0], 4 )}' )
weather_train.drop(['cloud_coverage','precip_depth_1_hr'], axis = 1,inplace=True)
weather_train['hour'] = weather_train.timestamp.dt.hour
weather_train['month'] = weather_train.timestamp.dt.month
def site_mean_weather(table):
    for col in list(table.columns[table.isnull().any()]):
        imputaion = table.groupby(['site_id','hour','month'])[col].transform('mean')
        table[col].fillna(imputaion,inplace = True)
    print('Imputation with mean values is completed.')
     
site_mean_weather(weather_train)
weather_train.isnull().sum()
weather_train.drop(['sea_level_pressure'], axis = 1, inplace = True)
gc.collect()
df = pd.merge(train,building, on="building_id", how="left")
df = df.merge(weather_train, on=["site_id", "timestamp"], how="left")
df.shape
df.head()
df.nunique()
df.isnull().sum()
df['month'] = df.timestamp.dt.month
df['hour'] = df.timestamp.dt.hour
site_mean_weather(df)
for col in list(df.columns[df.isnull().any()]):    
    imputaion = df.groupby(['hour','month'])[col].transform('mean')
    df[col].fillna(imputaion,inplace = True)
print('Imputation is completed.')
del train
del building
del weather_train
gc.collect()
df.dtypes
df[['primary_use','hour','month','site_id','building_id','wind_direction']] = df[['primary_use','hour','month','site_id','building_id','wind_direction']].astype('category')
gc.collect()
df['cons/sqft'] = df['meter_reading'] / df['square_feet']
df.loc[(df['site_id'] == 0) & (df['meter'] == 'electricity'), 'meter_reading'] = df[(df['site_id'] == 0) & (df['meter'] == 'electricity')]['meter_reading'].apply(lambda x: x* 0.2931 )
df['day'] = df.timestamp.dt.year
df = reduce_mem_usage(df)
gc.collect()
fig, axes = plt.subplots(1, 2, figsize=(12, 4),constrained_layout=True)
fig.suptitle('Building Counts')
sns.barplot(ax=axes[0],y="site_id", x='building_id', data=df.groupby(['site_id'])['building_id'].nunique().reset_index())
axes[0].set(xlabel = 'Building Count', ylabel = 'Site id')  
sns.barplot(ax=axes[1],y="primary_use", x='building_id', data=df.groupby(['primary_use'])['building_id'].nunique().reset_index())
axes[1].set(xlabel = 'Building Count', ylabel = 'Primary Use')
gc.collect()
fig, axes = plt.subplots(2, 2, figsize=(20, 16),constrained_layout=True)
fig.suptitle('Energy Consumption for Primary Use')
sns.boxplot(ax=axes[0, 0], y='primary_use', x='meter_reading', data=df, showfliers=True)
sns.boxplot(ax=axes[0, 1], y='primary_use', x='meter_reading', data=df, showfliers=False)
sns.boxplot(ax=axes[1, 0], y='primary_use', x='cons/sqft', data=df, showfliers=True)
sns.boxplot(ax=axes[1, 1], y='primary_use', x='cons/sqft', data=df, showfliers=False);
fig, axes = plt.subplots(2, 2, figsize=(14, 16),constrained_layout=True)
fig.suptitle('Energy Consumption for Sites')
sns.boxplot(ax=axes[0, 0], y='site_id', x='meter_reading', data=df, showfliers=True)
sns.boxplot(ax=axes[0, 1], y='site_id', x='meter_reading', data=df, showfliers=False)
sns.boxplot(ax=axes[1, 0], y='site_id', x='cons/sqft', data=df, showfliers=True)
sns.boxplot(ax=axes[1, 1], y='site_id', x='cons/sqft', data=df, showfliers=False);
gc.collect()
df['day'] = df.timestamp.dt.day
df[df['meter']=='electricity'].groupby(['site_id','month','day','building_id'])['meter_reading'].sum().sort_values(ascending = False).reset_index().head(100)
df[df['meter']=='electricity'].groupby(['site_id','month','day','building_id'])['cons/sqft'].sum().sort_values(ascending = False).reset_index().head(100)
df.groupby(['site_id'])['meter_reading'].sum().sort_values(ascending = False).reset_index()
gc.collect()
gc.collect()
total = 0
print('Outlier distribution in meter types')
for col in list(df['meter'].unique()):
    r = np.percentile(df[df['meter'] == col]['cons/sqft'],75) + 1.5 * (np.percentile(df[df['meter'] == col]['cons/sqft'],75) - np.percentile(df[df['meter'] == col]['cons/sqft'],25))
    print(f'''Percentage of outliers in {col} to all readings: { round(df[(df['meter'] == col) & (df['cons/sqft'] > r)].shape[0]/ df.shape[0],5)}''')
    total +=  df[(df['meter'] == col) & (df['cons/sqft'] > r)].shape[0]
print(f'Total fraction of outliers {round(total / df.shape[0],5)}')
gc.collect()
total = 0
print('Outlier distribution in building types')
for col in list(df['primary_use'].unique()):
    r = np.percentile(df[df['primary_use'] == col]['cons/sqft'],75) + 1.5 * (np.percentile(df[df['primary_use'] == col]['cons/sqft'],75) - np.percentile(df[df['primary_use'] == col]['cons/sqft'],25))
    print(f'''Percentage of outliers in {col} to all readings: { round(df[(df['primary_use'] == col) & (df['cons/sqft'] > r)].shape[0]/ df.shape[0],5)}''')
    total +=  df[(df['primary_use'] == col) & (df['cons/sqft'] > r)].shape[0]
print(f'Total fraction of outliers {round(total / df.shape[0],5)}')
gc.collect()
total = 0
print('Outlier distribution in building types')
for col in list(df['site_id'].unique()):
    r = np.percentile(df[df['site_id'] == col]['cons/sqft'],75) + 1.5 * (np.percentile(df[df['site_id'] == col]['cons/sqft'],75) - np.percentile(df[df['site_id'] == col]['cons/sqft'],25))
    print(f'''Percentage of outliers in {col} to all readings: { round(df[(df['site_id'] == col) & (df['cons/sqft'] > r)].shape[0]/ df.shape[0],5)}''')
    total +=  df[(df['site_id'] == col) & (df['cons/sqft'] > r)].shape[0]
print(f'Total fraction of outliers {round(total / df.shape[0],5)}')
gc.collect()
df.groupby(['month','day','building_id','meter'])['meter_reading'].sum().sort_values(ascending = False).reset_index().head()
df.groupby(['month','day','building_id','meter'])['meter_reading'].sum().sort_values(ascending = False).reset_index().head(100)['building_id'].value_counts()
gc.collect()
df.groupby(['building_id','site_id','meter'])['meter_reading'].sum().sort_values(ascending = False).reset_index().head(10)
df.groupby(['building_id','site_id','meter'])['cons/sqft'].sum().sort_values(ascending = False).reset_index().head(10)
df["weekday"] = df.timestamp.dt.weekday 
df.loc[df['weekday'].isin([5, 6]), 'Weekend'] = 1
df['Weekend'].fillna(0,inplace = True)
df['Weekend'] = df['Weekend'].astype('bool')