import numpy  as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import gc

import warnings

warnings.filterwarnings("ignore")



#将图表嵌入到notebook中

%matplotlib inline   
import os

print(os.listdir('/kaggle/input/ashrae-energy-prediction/'))
%%time

root='/kaggle/input/ashrae-energy-prediction/'

train_df=pd.read_csv(root+'train.csv')

#转换日期格式，方便以后的处理

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')

weather_train_df = pd.read_csv(root + 'weather_train.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')



weather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')

weather_test_df["timestamp"] = pd.to_datetime(weather_test_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



building_meta_df = pd.read_csv(root + 'building_metadata.csv')



test_df = pd.read_csv(root + 'test.csv')

test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)

print('Size of test_df data', test_df.shape)

print('Size of sample_submission data', sample_submission.shape)
train_df.head()
weather_train_df.head()
weather_test_df.head()
building_meta_df.head()
test_df.head()
sample_submission.head()
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
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)
plt.figure(figsize=(15,5))

train_df['meter_reading'].plot()
#set_index

train=train_df.set_index(['timestamp'])



#plot missing values per building/meter

f,a=plt.subplots(1,4,figsize=(20,30))

for meter in np.arange(4):

    df=train[train.meter==meter].copy().reset_index()

    df['timestamp']=pd.to_timedelta(df.timestamp-pd.to_datetime("2016-01-01")).dt.total_seconds()/3600

    df['timestamp']=df.timestamp.astype(int)

    df.timestamp=df.timestamp-df.timestamp.min()

    missmap=np.empty((1449,df.timestamp.max()+1))

    missmap.fill(np.nan)

    for l in df.values:

        #print(l)

        if l[2]!=meter:continue

        missmap[int(l[1]),int(l[0])]=0 if l[3]==0 else 1

    a[meter].set_title(f'meter {meter:d}')

    sns.heatmap(missmap,cmap='Paired',ax=a[meter],cbar=True)

        

        
total = train_df.isnull().sum().sort_values(ascending=False)

percent=(train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)

missing_train_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_train_data
# checking missing data

total = weather_train_df.isnull().sum().sort_values(ascending = False)

percent = (weather_train_df.isnull().sum()/weather_train_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_data.head(9)
# checking missing data

total = weather_test_df.isnull().sum().sort_values(ascending = False)

percent = (weather_test_df.isnull().sum()/weather_test_df.isnull().count()*100).sort_values(ascending = False)

missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_weather_data.head(9)
# checking missing data

total = building_meta_df.isnull().sum().sort_values(ascending = False)

percent = (building_meta_df.isnull().sum()/building_meta_df.isnull().count()*100).sort_values(ascending = False)

missing_building_meta_df  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_building_meta_df.head(9)
train_df=train_df.merge(building_meta_df,on='building_id',how='left')

test_df=test_df.merge(building_meta_df,on='building_id',how='left')



train_df=train_df.merge(weather_train_df,on=['site_id','timestamp'],how='left')

test_df=test_df.merge(weather_test_df,on=['site_id','timestamp'],how='left')





gc.collect()
fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)

train_df[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);

train_df[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);

axes.set_title('Mean Meter reading by hour and day', fontsize=16);



#显示图中的标签

axes.legend();
train_df.head()
fig, axes = plt.subplots(8,2,figsize=(15, 30), dpi=100)

for i in range(train_df['site_id'].nunique()):

    train_df[train_df['site_id'] == i][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

    train_df[train_df['site_id'] == i][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=1, label='By day', color='tab:orange').set_xlabel('');

    axes[i%8][i//8].legend();

    axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);

    plt.subplots_adjust(hspace=0.45)
fig, axes = plt.subplots(8,2,figsize=(14, 30), dpi=100)

for i, use in enumerate(train_df['primary_use'].value_counts().index.to_list()):

    try:

        train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == use)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

        train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == use)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=1, label='By day', color='tab:orange').set_xlabel('');

        axes[i%8][i//8].legend();

    except TypeError:

        pass

    axes[i%8][i//8].set_title(use, fontsize=13);

    plt.subplots_adjust(hspace=0.45)
fig, axes = plt.subplots(3,1,figsize=(14, 18), dpi=100)

for i in train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education')]['meter'].value_counts(dropna=False).index.to_list():

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == i)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == i)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i], alpha=1, label='By day', color='tab:orange').set_xlabel('');

    axes[i].legend();

    axes[i].set_title('Meter: ' + str(i), fontsize=13);
len(train_df[(train_df['primary_use']=='Education')&(train_df['site_id']==13)&(train_df['meter']==2)]['building_id'].value_counts().index.to_list())
fig, axes = plt.subplots(9,2,figsize=(14, 36), dpi=100)

for i, building in enumerate(train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2)]['building_id'].value_counts(dropna=False).index.to_list()):

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2) & (train_df['building_id'] == building)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%9][i//9], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

    train_df[(train_df['site_id'] == 13) & (train_df['primary_use'] == 'Education') & (train_df['meter'] == 2) & (train_df['building_id'] == building)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%9][i//9], alpha=1, label='By day', color='tab:orange').set_xlabel('');

    axes[i%9][i//9].legend();

    axes[i%9][i//9].set_title('building_id: ' + str(building), fontsize=13);

    plt.subplots_adjust(hspace=0.45)
fig, axes = plt.subplots(3,1,figsize=(14, 20), dpi=100)



train_df[(train_df['meter'] == 2) & (train_df['building_id'] == 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[0], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

train_df[(train_df['meter'] == 2) & (train_df['building_id'] == 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[0], alpha=1, label='By day', color='tab:orange').set_xlabel('');



train_df[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[1], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

train_df[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[1], alpha=1, label='By day', color='tab:orange').set_xlabel('');



train_df[~((train_df['meter'] == 2) & (train_df['building_id'] == 1099))][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[2], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

train_df[~((train_df['meter'] == 2) & (train_df['building_id'] == 1099))][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[2], alpha=1, label='By day', color='tab:orange').set_xlabel('');



axes[0].set_title('building_id==1099 and meter==2', fontsize=13);

axes[1].set_title('Full dataset', fontsize=13);

axes[2].set_title('building_id 1099 excluded', fontsize=13);

plt.subplots_adjust(hspace=0.45)
# Find correlations with the target and sort

correlations = train_df.corr()['meter_reading'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(7))

print('\nMost Negative Correlations:\n', correlations.head(4))



corrs = train_df.corr()

corrs
def plot_dist_col(column):

    fig,ax=plt.subplots(figsize=(10,10))

    sns.distplot(weather_train_df[column].dropna(),color='green',ax=ax).set_title(column,fontsize=16)

    sns.distplot(weather_test_df[column].dropna(),color='purple',ax=ax).set_title(column,fontsize=16)

    plt.xlabel(column,fontsize=15)

    plt.legend(['train','test'])

    plt.show()
plot_dist_col('air_temperature')
plot_dist_col('cloud_coverage')
weather_train_df['cloud_coverage'].unique()
plot_dist_col('dew_temperature')
plot_dist_col('precip_depth_1_hr')
weather_train_df['precip_depth_1_hr'].unique()
plot_dist_col('sea_level_pressure')
plot_dist_col('wind_direction')
plot_dist_col('wind_speed')
#风速小于0，这里不确定是数据有问题还是因为逆风

len(weather_train_df[weather_train_df['precip_depth_1_hr']<0])
def plot_dist_col(column):

    fig,ax=plt.subplots(figsize=(10,10))

    sns.distplot(building_meta_df[column].dropna(),color='green',ax=ax).set_title(column,fontsize=16)

       

    plt.xlabel(column,fontsize=15)

    plt.legend(['building'])

    plt.show()
plot_dist_col('floor_count')
plot_dist_col('year_built')
building_meta_df['year_built'].unique()
plot_dist_col('square_feet')
train_df_sub_1099=train_df[~((train_df['meter'] == 2) & (train_df['building_id'] == 1099))]

ts=train_df_sub_1099.groupby(["timestamp"])["meter_reading"].mean()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('average meter_reading with time')

plt.xlabel('timestamp')

plt.ylabel('meter_reading')

plt.plot(ts)
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12).mean(),label='Rolling Mean')

plt.plot(ts.rolling(window=12).std(),label='Rolling sd')

plt.legend()
building_meta_df.groupby('site_id').primary_use.agg(lambda x:x.value_counts().to_dict()).to_dict()
building_meta_df.groupby('site_id').building_id.agg(lambda x:x.value_counts().to_dict()).to_dict()[0]
fig, axes = plt.subplots(8,2,figsize=(14, 20), dpi=100)

#weather_train_df[weather_train_df['site_id']==0].plot()



import datetime

# plt.figure(figsize=(16,6))



# plt.xlabel('timestamp')

# plt.ylabel('air_temperature')



#plt.plot(weather_train_df[])

#plt.legend()

weather_train_df['hour']=weather_train_df.timestamp.dt.hour

weather_train_df_mean_by_hour=weather_train_df.groupby(['site_id','hour']).mean()

# #train_df[train_df['site_id'] == i][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13);

for i in range(16):

    weather_train_df_mean_by_hour[i*24:(i+1)*24]['air_temperature'].plot(ax=axes[i%8][i//8])

    axes[i%8][i//8].legend();

    axes[i%8][i//8].set_title('site_id {} max temperature hour is {}'.format(i,np.argmax(weather_train_df_mean_by_hour[i*24:(i+1)*24]['air_temperature'])[1]), fontsize=13);

plt.subplots_adjust(hspace=0.8)
#load training data 2016

weather_train=pd.read_csv(root+'weather_train.csv', parse_dates=['timestamp'])

weather_train.head()
#pivot to plot

wmatrix_train=weather_train.pivot(index='timestamp',columns='site_id',values='air_temperature')

wmatrix_train.head()
#寻找训练集中有最多缺失值的site_id

site_id=wmatrix_train.count().idxmin()



#挑选出一段时间进行观测

start_date,end_date=datetime.date(2016,1,1),datetime.date(2016,1,9)



#初始化绘图

f,ax=plt.subplots(figsize=(18,6))



#load test data 2017-2018

weather_test=pd.read_csv(root+'weather_test.csv', parse_dates=['timestamp'])



#shift 2017 to 2016

weather_test.timestamp=weather_test.timestamp-datetime.timedelta(365)

wtmatrix=weather_test.pivot(index='timestamp',columns='site_id',values='air_temperature')

wtmatrix.loc[start_date:end_date,site_id].plot(ax=ax,label=f'2017.1.1-2017.1.9 site:{site_id}',alpha=0.5)



#shift 2018 to 2016

weather_test.timestamp=weather_test.timestamp-datetime.timedelta(365)

wtmatrix=weather_test.pivot(index='timestamp',columns='site_id',values='air_temperature')

wtmatrix.loc[start_date:end_date,site_id].plot(ax=ax,label=f'2018.1.1.-2018.1.9 site:{site_id}',alpha=0.5)





def fill_with_polynomial(wmatrix):

    return wmatrix.fillna(wmatrix.interpolate(method='polynomial',order=3))



def fill_with_lin(wmatrix):

    return wmatrix.fillna(wmatrix.interpolate(method='linear'))



def fill_with_mix(wmatrix):

    wmatrix=(wmatrix.fillna(wmatrix.interpolate(method='linear',limit_direction='both'))+ wmatrix.fillna(wmatrix.interpolate(method='polynomial', order=3, limit_direction='both')))*0.5  

    

    return wmatrix        # fill with second item

    

fill_with_lin(wmatrix_train).loc[start_date:end_date,site_id].plot(ax=ax,label=f'linear Jan 2016 site:{site_id}',alpha=0.5) 

fill_with_polynomial(wmatrix_train).loc[start_date:end_date,site_id].plot(ax=ax,label=f'polynomial Jan 2016 site:{site_id}',alpha=0.5)    

fill_with_mix(wmatrix_train).loc[start_date:end_date,site_id].plot(ax=ax,label=f'mix Jan 2016 site:{site_id}',alpha=0.5)    

wmatrix_train.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2016 site:{site_id}')

plt.legend()
# pivot to plot

col = 'dew_temperature'

wmatrix = weather_train.pivot(index='timestamp', columns='site_id', values=col)

# site with largest amount of missing data points

site_id = wmatrix.count().idxmin()

# plot perid

start_date, end_date = datetime.date(2016, 1, 1), datetime.date(2016, 1, 12)

f,ax = plt.subplots(figsize=(18,6))



_ = fill_with_lin(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'linear Jan 2016 site:{site_id}', alpha=0.5)

_ = fill_with_polynomial(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'cubic Jan 2016 site:{site_id}', alpha=0.5)

_ = fill_with_mix(wmatrix).loc[start_date:end_date, site_id].plot(ax=ax, label=f'mix Jan 2016 site:{site_id}', alpha=0.5)

_ = wmatrix.loc[start_date:end_date, site_id].plot(ax=ax, label=f'Jan 2016 site:{site_id}')



_ = plt.legend()