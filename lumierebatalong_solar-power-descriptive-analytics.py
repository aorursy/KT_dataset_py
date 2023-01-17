# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import all package needed

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import normaltest

import holoviews as hv

from holoviews import opts

import cufflinks as cf

hv.extension('bokeh')
file1 = '/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv'

file2 = '/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv'

file3 = '/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv'

file4 = '/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv'
plant1 = pd.read_csv(file1)

sensor1 = pd.read_csv(file3)
plant2 = pd.read_csv(file2)

sensor2 = pd.read_csv(file4)
plant1.tail()
sensor1.tail()
plant1.info()
sensor1.info()
plant2.tail()
sensor2.tail()
plant2.info()
sensor2.info()
#how many inverters we have in plant I and II

print('We have: \n 1- For plant I: {} Inverters. \n 2- for Plant II: {} Inverters.'.format(plant1['SOURCE_KEY'].nunique(),

                                                                                         plant2['SOURCE_KEY'].nunique()))
plant1.drop(columns = 'PLANT_ID').describe()
sensor1.drop(columns = 'PLANT_ID').describe()
fig = plt.figure(dpi=100, figsize=(15,10))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

cols = list(set(plant1.columns) - set(['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME']))

for i in range(1,5):

    ax = fig.add_subplot(2,2,i)

    sns.violinplot(plant1[cols[i-1]] , ax=ax)
fid = plt.figure(dpi=100, figsize=(15,10))

fid.subplots_adjust(wspace=0.2, hspace=0.2)

cls = list(set(sensor1.columns) - set(['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME']))

for i in range(1,4):

    ax = fid.add_subplot(2,2,i)

    sns.violinplot(sensor1[cls[i-1]] , ax=ax)
def hist2D(df = None, col1 = None, col2 = None, xlabel = None, ylabel = None):

    '''

        df: DataFrame

        col1, col2: columns from DataFrame

        xlabel,ylabe for name for plotting

    '''

    plt.figure(figsize=(15,5))

    plt.hist2d(df[col1], df[col2], bins = (30, 30))

    plt.colorbar()

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    ax = plt.gca()

    ax.axis('tight')
plant2.drop(columns = 'PLANT_ID').describe()
sensor2.drop(columns = 'PLANT_ID').describe()
figu = plt.figure(dpi=100, figsize=(15,10))

figu.subplots_adjust(wspace=0.2, hspace=0.2)

for i in range(1,5):

    ax = figu.add_subplot(2,2,i)

    sns.violinplot(plant2[cols[i-1]] , ax=ax)
fis = plt.figure(dpi=100, figsize=(15,10))

fis.subplots_adjust(wspace=0.2, hspace=0.2)

for i in range(1,4):

    ax = fis.add_subplot(2,2,i)

    sns.violinplot(sensor2[cls[i-1]] , ax=ax)
#convert date time object type to datetime

plant1['DATE_TIME'] = pd.to_datetime(plant1.pop('DATE_TIME'), format='%d-%m-%Y %H:%M')

plant2['DATE_TIME'] = pd.to_datetime(plant2.pop('DATE_TIME'), format='%Y-%m-%d %H:%M')

sensor2['DATE_TIME'] = pd.to_datetime(sensor2.pop('DATE_TIME'), format='%Y-%m-%d %H:%M')

sensor1['DATE_TIME'] = pd.to_datetime(sensor1.pop('DATE_TIME'), format='%Y-%m-%d %H:%M')
#I remove time in Date Time to get only date.

plant1['DATE'] = plant1.DATE_TIME.dt.date

plant2['DATE'] = plant2.DATE_TIME.dt.date
mean_daily_yield1 = plant1.groupby(by='DATE')['DAILY_YIELD'].agg('mean').reset_index()

mean_daily_yield2 = plant2.groupby(by='DATE')['DAILY_YIELD'].agg('mean').reset_index()
## we plot a mean
plt.figure(figsize=(15,5))

sns.lineplot(x='DATE', y='DAILY_YIELD', data=mean_daily_yield1)

plt.grid(True)

plt.title('Mean Daily Yield for Plant I.',  weight='bold')

plt.ylabel('MEAN DAILY YIELD')

plt.ylim(2000,5500)

plt.show()
plt.figure(figsize=(15,5))

sns.lineplot(x='DATE', y='DAILY_YIELD', data=mean_daily_yield2)

plt.grid(True)

plt.title('Mean Daily Yield for Plant II.',  weight='bold')

plt.ylabel('MEAN DAILY YIELD')

plt.ylim(1500,4500)

plt.show()
mean = pd.DataFrame()

mean['Mean_Daily_Yield_PLANTI'] = mean_daily_yield1.mean()

mean['Mean_Daily_Yield_PLANTII'] = mean_daily_yield2.mean()
mean.T.style.background_gradient('viridis')
print('Gap between two plants for mean daily yield is {} KWh.'.\

      format(round(abs(mean.Mean_Daily_Yield_PLANTI.values[0] -

                                                            mean.Mean_Daily_Yield_PLANTII.values[0]),2)))
mean.T.plot(kind='pie', subplots=True, figsize=(15,10))

plt.title('Mean Daily Yield Comparison',  weight='bold')

plt.show()
sensor1['DATE'] = sensor1.DATE_TIME.dt.date

sensor2['DATE'] = sensor2.DATE_TIME.dt.date
total_irradiation1 = sensor1.groupby('DATE')['IRRADIATION'].agg('sum').reset_index()

total_irradiation2 = sensor2.groupby('DATE')['IRRADIATION'].agg('sum').reset_index()
plt.figure(figsize=(15,5))

sns.lineplot(x='DATE', y='IRRADIATION', data=total_irradiation1)

plt.grid(True)

plt.title('TOTAL IRRADIATION PER DAY FOR PLANT I.',  weight='bold')

plt.ylim(10,30)

plt.show()
plt.figure(figsize=(15,5))

sns.lineplot(x='DATE', y='IRRADIATION', data=total_irradiation2)

plt.grid(True)

plt.title('TOTAL IRRADIATION PER DAY FOR PLANT II.',  weight='bold')

plt.ylim(10,30)

plt.show()
temp = plt.figure(figsize=(20,5), dpi=100)

temp.subplots_adjust(wspace=0.1)

ax3 = temp.add_subplot(1,2,1)

ax4 = temp.add_subplot(1,2,2)

sns.lineplot(x='DATE', y='AMBIENT_TEMPERATURE', data=sensor1, ax=ax3)

sns.lineplot(x='DATE', y='MODULE_TEMPERATURE', data=sensor1, ax=ax4)

ax3.set_title('TEMPERATURE FOR PLANT I',  weight='bold')

ax4.set_title('TEMPERATURE FOR PLANT I',  weight='bold')

ax3.grid(True)

ax4.grid(True)

plt.show()
te = plt.figure(figsize=(20,5), dpi=100)

te.subplots_adjust(wspace=0.1)

ax5 = te.add_subplot(1,2,1)

ax6 = te.add_subplot(1,2,2)

sns.lineplot(x='DATE', y='AMBIENT_TEMPERATURE', data=sensor2, ax=ax5)

sns.lineplot(x='DATE', y='MODULE_TEMPERATURE', data=sensor2, ax=ax6)

ax5.set_title('TEMPERATURE FOR PLANT II',  weight='bold')

ax6.set_title('TEMPERATURE FOR PLANT II',  weight='bold')

ax5.grid(True)

ax6.grid(True)

plt.show()
temp_plant1 = pd.DataFrame(sensor1[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].max(), columns=['PLANT I'])

temp_plant2 = pd.DataFrame(sensor2[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].max(), columns=['PLANT II'])
temp_plant1.style.background_gradient('viridis')
temp_plant2.style.background_gradient('viridis')
pie = plt.figure(figsize=(20,10))

pie.subplots_adjust(wspace=0.2)

ax7 = pie.add_subplot(1,2,1)

ax8 = pie.add_subplot(1,2,2)



temp_plant1.plot(kind='pie', subplots=True, ax=ax7)

temp_plant2.plot(kind='pie', subplots=True, ax=ax8)

ax7.set_title('Plant I Max Temperature', weight='bold')

ax8.set_title('Plant II Max Temperature',  weight='bold')

plt.show()
print('Plant I have {} inverters.'.format(plant1['SOURCE_KEY'].nunique()))
print('Plant II have {} inverters.'.format(plant2['SOURCE_KEY'].nunique()))
#the data have been recorded after 15min. But we are transforming it for 1h

plant1_group = plant1.groupby('DATE_TIME')[['AC_POWER', 'DC_POWER']].agg('sum')
# slice [start:stop:step], starting from index 4 take every 5th record.

plant1_group =  plant1_group[0::4].reset_index()

plant1_group['Date'] = plant1_group.DATE_TIME.dt.date
date1 = plant1_group.Date.unique()
maximun1 = []

minimun1 = []



for dt in date1:

    maximun1.append(plant1_group[plant1_group.Date==dt].max())

    minimun1.append(plant1_group[plant1_group.Date==dt].min())
min_plant1 = pd.DataFrame(minimun1)

max_plant1 = pd.DataFrame(maximun1)
min_plant1
max_plant1.style.background_gradient('viridis')
#the data have been recorded after 15min. But we are transforming it for 1h

plant2_group = plant2.groupby('DATE_TIME')[['AC_POWER', 'DC_POWER']].agg('sum')
# slice [start:stop:step], starting from index 4 take every 5th record.

plant2_group =  plant2_group[0::4].reset_index()

plant2_group['Date'] = plant2_group.DATE_TIME.dt.date
date2 = plant2_group.Date.unique()
maximun2 = []

minimun2 = []



for dt in date2:

    maximun2.append(plant2_group[plant2_group.Date==dt].max())

    minimun2.append(plant2_group[plant2_group.Date==dt].min())
min_plant2 = pd.DataFrame(minimun2)

max_plant2 = pd.DataFrame(maximun2)
min_plant2
max_plant2.style.background_gradient('viridis')
inverter1 = plant1.groupby('SOURCE_KEY')[['AC_POWER', 'DC_POWER']].agg('sum')

inverter2 = plant2.groupby('SOURCE_KEY')[['AC_POWER', 'DC_POWER']].agg('sum')
inverter1.plot(kind='bar', subplots=True, figsize=(20,15))

plt.show()
stop1 = inverter1 == inverter1.max()
print('The inverter  has produced maximun DC/AC POWER for plant I is: {}'.format(inverter1.index[stop1.iloc[:,0]].values[0]))
inverter2.plot(kind='bar', subplots=True, figsize=(20,15))

plt.show()
stop2 = inverter2 == inverter2.max()
print('The inverter  has produced maximun DC/AC POWER for plant II is: {}'.format(inverter2.index[stop2.iloc[:,0]].values[0]))
inverter1.sort_values(by=['AC_POWER'], ascending=False).style.background_gradient('viridis')
inverter2.sort_values(by=['AC_POWER'], ascending=False).style.background_gradient('viridis')