# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from datetime import timedelta

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]

train = pd.read_csv( "../input/ashrae-energy-prediction/train.csv")

test = pd.read_csv( "../input/ashrae-energy-prediction/test.csv")

building = pd.read_csv( "../input/ashrae-energy-prediction/building_metadata.csv")

weather_train= pd.read_csv( "../input/ashrae-energy-prediction/weather_train.csv")
train['timestamp'] = pd.to_datetime(train['timestamp'],format='%Y-%m-%d %H:%M:%S' )

test['timestamp'] = pd.to_datetime(test['timestamp'],format='%Y-%m-%d %H:%M:%S' )

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'],format='%Y-%m-%d %H:%M:%S' )
print('Number of buildings included in  train is', train.building_id.nunique() ,',starting with id number',train.building_id.min(),'and ending with id number',train.building_id.max())

print('Number of buildings included in  test is', test.building_id.nunique() ,',starting with id number',test.building_id.min(),'and ending with id number',test.building_id.max()  )
def vertlabel(plot):

    for p in plot.patches[0:]:

        h = p.get_height()

        x = p.get_x()+p.get_width()/2.

        if h != 0:

            plot.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                           textcoords="offset points", ha="center", va="bottom")
df = train.groupby(['building_id','meter']).size().reset_index().rename(columns={0:'count'})

df.drop('count',axis=1,inplace=True)

df_meter = df.groupby(['meter']).size().reset_index().rename(columns={0:'count'})

df_build = df.groupby(['building_id']).size().reset_index().rename(columns={0:'count'})

build_meter= df_build['count'].value_counts()

sns.set(rc={'figure.figsize':(12,6)})

plt.subplot(1,2,1)

bar1=sns.barplot(x=df_meter.meter,y= df_meter['count'])

bar1.set(ylim=(0,1600),title= "Count of Buildings Vs type of meter")

vertlabel(bar1)

plt.subplot(1,2,2)

bar2 = sns.barplot(x=build_meter.index,y= build_meter.values)

bar2.set(ylim=(0,1000),title= "Count of Buildings Vs Number of types of  installed meter",xlabel= "Number of types of  installed meter")

vertlabel(bar2)

plt.tight_layout(pad=2, w_pad=5, h_pad=1.0)
len(df)
print('Start time for the time series  in train dataset is :',train.timestamp.min())

print('End time for the time series in train dataset is :',train.timestamp.max())

print('Number of time points for the time series  in train dataset is :',train.timestamp.nunique())

print('Start time for the time series  in test dataset is :',test.timestamp.min())

print('End time for the time series in test dataset is :',test.timestamp.max())

print('Number of time points for the time series  in test dataset is :',test.timestamp.nunique())
df_read=train.groupby(['building_id','meter']).size().value_counts().rename(columns={0:'reading_count'})

bar3= sns.barplot(x= df_read.index[:20],y=df_read.values[:20])

bar3.set(title='Data points for House- Meter combination - Top Twenty datapoints')

vertlabel(bar3)
df_time = train.groupby(['building_id','meter']).agg({'timestamp':[min,max,'count']})

df_time.columns = ["_".join(x) for x in df_time.columns.ravel()]

df_time = df_time.reset_index()

print('Number of different starting times ',df_time.timestamp_min.nunique())

print('Number of different ending times ',df_time.timestamp_max.nunique())
pd.crosstab(df_time[df_time.timestamp_max==df_time.timestamp_max.max()]['timestamp_min'],df_time[df_time.timestamp_max==df_time.timestamp_max.max()]['timestamp_max'],margins=True)
pd.crosstab(df_time[df_time.timestamp_max<df_time.timestamp_max.max()]['timestamp_min'],df_time[df_time.timestamp_max<df_time.timestamp_max.max()]['timestamp_max'],margins=True)
datetimeFormat = '%Y-%m-%d %H:%M:%S'

df_time['Expected_count'] = (df_time['timestamp_max']- df_time['timestamp_min']).astype('timedelta64[h]')+1

df_time['missing_time_points']= df_time['Expected_count'] - df_time['timestamp_count']

df_time['percent_missing_time_points']= ((df_time['missing_time_points']/df_time['Expected_count'])*100).round(2)

df_time.head()
sns.set(rc={'figure.figsize':(10,4)})

full_data= df_time[df_time.percent_missing_time_points==0]['meter'].value_counts()

per99_data = df_time[(df_time.percent_missing_time_points>0)& (df_time.percent_missing_time_points<1)]['meter'].value_counts()

data = pd.DataFrame({'meter':full_data.index, 'No_missing' : full_data.values,'one_percent_missing' : per99_data.values })

bar4 = data.set_index('meter').T.plot(kind='bar', stacked=True)
df_time[df_time.percent_missing_time_points>20]
df_time_missing = df_time[(df_time.percent_missing_time_points>1) & (df_time.percent_missing_time_points<21)]

meter0=df_time_missing.percent_missing_time_points[df_time_missing.meter==0]

meter1=df_time_missing.percent_missing_time_points[df_time_missing.meter==1]

meter2=df_time_missing.percent_missing_time_points[df_time_missing.meter==2]

meter3=df_time_missing.percent_missing_time_points[df_time_missing.meter==3]

hist1=plt.hist([meter0,meter1,meter2,meter3], label=['meter0','meter1','meter2','meter3'],stacked=True)

plt.legend(loc='upper right')

plt.show()
df_time_test = test.groupby(['building_id','meter']).agg({'timestamp':[min,max,'count']})

df_time_test.columns = ["_".join(x) for x in df_time_test.columns.ravel()]

df_time_test = df_time_test.reset_index()

print('Number of different starting times ',df_time_test.timestamp_min.nunique())

print('Number of different ending times ',df_time_test.timestamp_max.nunique())
df_time_test['Expected_count'] = (df_time_test['timestamp_max']- df_time_test['timestamp_min']).astype('timedelta64[h]')+1

df_time_test['missing_time_points']= df_time_test['Expected_count'] - df_time_test['timestamp_count']

df_time_test['percent_missing_time_points']= ((df_time_test['missing_time_points']/df_time_test['Expected_count'])*100).round(2)

df_time_test.head()
len(df_time_test[df_time.missing_time_points>0])
building.head()
sns.set(rc={'figure.figsize':(12,6)})

count1= sns.countplot(data=building,x='site_id')

count1.set(title= 'Number of Buildings site wise')

vertlabel(count1)
sns.set(rc={'figure.figsize':(12,6)})

from textwrap import wrap

labels = building['primary_use'].unique()

labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

count2= sns.countplot(data=building,x='primary_use')

count2.set(title= 'Number of Buildings primary use wise')

count2.set_xticklabels(labels,rotation=90)

vertlabel(count2)
from matplotlib.colors import ListedColormap

sns.set(rc={'figure.figsize':(12,6)})

sns.set_style("whitegrid")

#qualitative_colors = sns.color_palette("Set3", 16)

my_cmap = ListedColormap(sns.color_palette("Paired", 16).as_hex())

df_plot = building.groupby(['site_id', 'primary_use']).size().reset_index().pivot(columns='primary_use', index='site_id', values=0)

plot1= df_plot.plot(kind='bar', stacked=True,colormap=my_cmap)

plot1.set(title= "Buildings by site and primary use")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
label=(25000,50000,100000,150000,200000,300000,400000,600000,800000)

box1=sns.boxplot(data=building,y="square_feet",x='primary_use')

box1.set_xticklabels(box1.get_xticklabels(),rotation=90,fontsize=10)

box1.set_yticks(label)

box1.set_title('Building Area by Primary use')
df_reading = train.groupby(['building_id','meter']).agg({'meter_reading':[min,max,'mean','count']})

df_reading.columns = ["_".join(x) for x in df_reading.columns.ravel()]

df_reading = df_reading.reset_index()
df_zero=train[train['meter_reading']==0].groupby(['building_id','meter']).size().reset_index().rename(columns={0:'zero_count'})

df_reading= df_reading.merge(df_zero,on=['building_id','meter'],how='left')

df_reading['zero_count'].fillna(0,inplace=True)

df_reading['zero_count_per'] = ((df_reading['zero_count']/df_reading['meter_reading_count'])*100).round(2)

                                
#df_reading.zero_count_per[df_reading.zero_count_per>0].hist()
df_reading= df_reading.merge(df_time,on=['building_id','meter'],how='left')

df_reading= df_reading.merge(building,on=['building_id'],how='left')
df_reading['total_per'] = df_reading['percent_missing_time_points'] + df_reading['zero_count_per']
hist1=plt.hist(df_reading['total_per'][df_reading['total_per']>5],bins=(5,10,20,30,50,75,100)) 

g= sns.FacetGrid(df_reading,col='meter',row='primary_use',margin_titles=True,sharex=False,sharey=False)

g.map(sns.scatterplot,'total_per','zero_count_per')

plt.tight_layout()
# Code for generating sample rows within groups

from random import sample

df_reading.groupby('primary_use',group_keys=False).apply(lambda x : x.sample(min(len(x),2)))
building_type_details=df_reading.groupby(['primary_use','meter']).agg({'meter_reading_mean': 'mean','square_feet':'mean','building_id':'count'})

building_type_details=building_type_details.reset_index()

building_type_details.rename(columns={'square_feet' :'avg_sq_feet','building_id': 'building_count'},inplace= True)

for column in ('meter_reading_mean','avg_sq_feet'):

    building_type_details[column]=building_type_details[column].astype(int)

building_type_details['meter'] = building_type_details['meter'].map({0: 'electricity', 1: 'chilledwater', 2: 'steam',3: 'hotwater'})
from textwrap import wrap

labels = building_type_details['primary_use'].unique()

labels = [ '\n'.join(wrap(l, 15)) for l in labels ]

g= sns.catplot(x='primary_use',y='meter_reading_mean',row='meter',data=building_type_details,kind='bar',aspect=4,sharex=False,sharey=False)

(g.set_xticklabels(labels,rotation=90,fontsize=12))

plt.tight_layout(pad=2, w_pad=5, h_pad=1.0)

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Average Meter readings by primary use of buildings')
overall = train.merge(building,how='left',on='building_id')

overall = overall.merge(weather_train,how='left',on=['site_id','timestamp'])
overall['month'] = overall.timestamp.dt.month

overall['day_month']= overall.timestamp.dt.day

overall['week']= overall.timestamp.dt.week
site_weather= overall.groupby(['site_id','month','day_month']).agg({'air_temperature' :(min,max,'mean'),

                                                          'dew_temperature' :(min,max,'mean'),

                                                          'precip_depth_1_hr' :(min,max,'mean'),

                                                          'sea_level_pressure' :(min,max,'mean'),

                                                         'wind_speed' :(min,max,'mean')})

site_weather.columns = ["_".join(x) for x in site_weather.columns.ravel()]

site_weather = site_weather.reset_index()
g= sns.FacetGrid(site_weather[site_weather.site_id<6],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','air_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','air_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','air_temperature_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of air temperature for Site id 0-5')
g= sns.FacetGrid(site_weather[(site_weather.site_id>5) & (site_weather.site_id<11)],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','air_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','air_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','air_temperature_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of air temperature for Site id 6-10')
g= sns.FacetGrid(site_weather[site_weather.site_id>10],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','air_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','air_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','air_temperature_max',color='red')

g.set_fontsize=14

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of air temperature for Site id 11-15')
g= sns.FacetGrid(site_weather[site_weather.site_id<6],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','dew_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','dew_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','dew_temperature_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of dew temperature for Site id 0-5')
g= sns.FacetGrid(site_weather[(site_weather.site_id>5) & (site_weather.site_id<11)],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','dew_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','dew_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','dew_temperature_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of dew temperature for Site id 6-10')
g= sns.FacetGrid(site_weather[site_weather.site_id>10],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','dew_temperature_min',color='lime')

g.map(sns.lineplot,'day_month','dew_temperature_mean',color='gold')

g.map(sns.lineplot,'day_month','dew_temperature_max',color='red')

g.set_fontsize=14

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of dew temperature for Site id 11-15')
g= sns.FacetGrid(site_weather[site_weather.site_id<6],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_min',color='lime')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_mean',color='gold')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of precipitation depth for Site id 0-5')
g= sns.FacetGrid(site_weather[(site_weather.site_id>5) & (site_weather.site_id<11) ],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_min',color='lime')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_mean',color='gold')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of precipitation depth for Site id 6-10')
g= sns.FacetGrid(site_weather[site_weather.site_id>10],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_min',color='lime')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_mean',color='gold')

g.map(sns.lineplot,'day_month','precip_depth_1_hr_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of precipitation depth for Site id 11-15')
g= sns.FacetGrid(site_weather[site_weather.site_id<6],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','sea_level_pressure_min',color='lime')

g.map(sns.lineplot,'day_month','sea_level_pressure_mean',color='gold')

g.map(sns.lineplot,'day_month','sea_level_pressure_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of sea level pressure for Site id 0-5')
g= sns.FacetGrid(site_weather[(site_weather.site_id>5) & (site_weather.site_id<11)],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','sea_level_pressure_min',color='lime')

g.map(sns.lineplot,'day_month','sea_level_pressure_mean',color='gold')

g.map(sns.lineplot,'day_month','sea_level_pressure_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of sea level pressure for Site id 6-10')
g= sns.FacetGrid(site_weather[site_weather.site_id>10],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','sea_level_pressure_min',color='lime')

g.map(sns.lineplot,'day_month','sea_level_pressure_mean',color='gold')

g.map(sns.lineplot,'day_month','sea_level_pressure_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of sea level pressure for Site id 11-15')
g= sns.FacetGrid(site_weather[site_weather.site_id<6],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','wind_speed_min',color='lime')

g.map(sns.lineplot,'day_month','wind_speed_mean',color='gold')

g.map(sns.lineplot,'day_month','wind_speed_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of Wind speed for Site id 0-5')
g= sns.FacetGrid(site_weather[(site_weather.site_id>5) & (site_weather.site_id<11)],col='site_id',row='month',margin_titles=True,sharey='col') 

g.map(sns.lineplot,'day_month','wind_speed_min',color='lime')

g.map(sns.lineplot,'day_month','wind_speed_mean',color='gold')

g.map(sns.lineplot,'day_month','wind_speed_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of Wind speed for Site id 6-10')
g= sns.FacetGrid(site_weather[site_weather.site_id>10],col='site_id',row='month',margin_titles=True,sharey='col')

g.map(sns.lineplot,'day_month','wind_speed_min',color='lime')

g.map(sns.lineplot,'day_month','wind_speed_mean',color='gold')

g.map(sns.lineplot,'day_month','wind_speed_max',color='red')

plt.tight_layout()

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Daily distribution of Wind speed for Site id 11-5')