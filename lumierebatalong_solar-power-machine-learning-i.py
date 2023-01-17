# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

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
cf.set_config_file(offline = True)

sns.set(style="whitegrid")
#we take file for plant 1 Generation data

file = '/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv'
plant1_data = pd.read_csv(file) # load data
plant1_data.tail()
print('The number of inverter for data_time {} is {}'.format('15-05-2020 23:00', plant1_data[plant1_data.DATE_TIME == '15-05-2020 23:00']['SOURCE_KEY'].nunique()))
plant1_data.info() # we check if there exist missing value
#we compute a sum of 22 inverters

plant1_data = plant1_data.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum')
plant1_data = plant1_data.reset_index()
plant1_data.head()
plant1_data['DATE_TIME'] = pd.to_datetime(plant1_data['DATE_TIME'], errors='coerce')
plant1_data['time'] = plant1_data['DATE_TIME'].dt.time

plant1_data['date'] = pd.to_datetime(plant1_data['DATE_TIME'].dt.date)
plant1_data.shape # our data reduced very well
#we check

plant1_data.head()
plant1_data.info()
#plant1_data.iplot(x= 'time', y='DC_POWER', xTitle='Time',  yTitle= 'DC Power', title='DC POWER plot')

plant1_data.plot(x= 'time', y='DC_POWER', style='.', figsize = (15, 8))

plant1_data.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.ylabel('DC Power')

plt.title('DC POWER plot')

plt.show()
#Okay, we are going to see dc power in each day produced by Plant.

#we create calendar_dc data how in each day Plant produce a dc power in each time.



calendar_dc = plant1_data.pivot_table(values='DC_POWER', index='time', columns='date')
calendar_dc.tail()
# define function to multi plot



def multi_plot(data= None, row = None, col = None, title='DC Power'):

    cols = data.columns # take all column

    gp = plt.figure(figsize=(20,20)) 

    

    gp.subplots_adjust(wspace=0.2, hspace=0.8)

    for i in range(1, len(cols)+1):

        ax = gp.add_subplot(row,col, i)

        data[cols[i-1]].plot(ax=ax, style = 'k.')

        ax.set_title('{} {}'.format(title, cols[i-1]))
multi_plot(data=calendar_dc, row=9, col=4)
daily_dc = plant1_data.groupby('date')['DC_POWER'].agg('sum')
daily_dc.plot.bar(figsize=(15,5), legend=True)

plt.title('Daily DC Power')

plt.show()
plant1_data.plot(x='time', y='DAILY_YIELD', style='b.', figsize=(15,5))

plant1_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('DAILY YIELD')

plt.ylabel('Yield')

plt.show()
#pivot table data

daily_yield = plant1_data.pivot_table(values='DAILY_YIELD', index='time', columns='date')
# we plot all daily yield

multi_plot(data=daily_yield.interpolate(), row=9, col=4, title='DAILY YIELD')
#plotting a change rate daily yield over time

multi_plot(data=daily_yield.diff()[daily_yield.diff()>0], row=9, col=4, title='new yield')
daily_yield.boxplot(figsize=(18,5), rot=90, grid=False)

plt.title('DAILY YIELD IN EACH DAY')

plt.show()
daily_yield.diff()[daily_yield.diff()>0].boxplot(figsize=(18,5), rot=90, grid=False)

plt.title('DAILY YIELD CHANGE RATE EACH 15 MIN EACH DAY')

plt.show()
#we compute a daily yield for each date.

dyield = plant1_data.groupby('date')['DAILY_YIELD'].agg('sum')
dyield.plot.bar(figsize=(15,5), legend=True)

plt.title('Daily YIELD')

plt.show()
file1 = '/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv'
plant1_sensor = pd.read_csv(file1)
plant1_sensor.head()
plant1_sensor.info()
plant1_sensor['DATE_TIME'] = pd.to_datetime(plant1_sensor['DATE_TIME'], errors='coerce')
# same work cleaning data

plant1_sensor['date'] = pd.to_datetime(pd.to_datetime(plant1_sensor['DATE_TIME']).dt.date)

plant1_sensor['time'] = pd.to_datetime(plant1_sensor['DATE_TIME']).dt.time





del plant1_sensor['PLANT_ID']

del plant1_sensor['SOURCE_KEY']
plant1_sensor.tail()
plant1_sensor.plot(x='time', y = 'AMBIENT_TEMPERATURE' , style='b.', figsize=(15,5))

plant1_sensor.groupby('time')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('Daily AMBIENT TEMPERATURE MEAN (RED)')

plt.ylabel('Temperature (°C)')

plt.show()
ambient = plant1_sensor.pivot_table(values='AMBIENT_TEMPERATURE', index='time', columns='date')
ambient.tail()
ambient.boxplot(figsize=(15,5), grid=False, rot=90)

plt.title('AMBIENT TEMPERATURE BOXES')

plt.ylabel('Temperature (°C)')
am_temp = plant1_sensor.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')
am_temp.plot(grid=True, figsize=(15,5), legend=True, colormap='Oranges_r')

plt.title('AMBIENT TEMPERATURE 15 MAY- 17 JUNE')

plt.ylabel('Temperature (°C)')
am_change_temp = (am_temp.diff()/am_temp)*100
am_change_temp.plot(figsize=(15,5), grid=True, legend=True)

plt.ylabel('%change')

plt.title('AMBIENT TEMPERATURE %change')
from scipy.signal import periodogram
decomp = sm.tsa.seasonal_decompose(am_temp)
cols = ['trend', 'seasonal', 'resid'] # take all column

data = [decomp.trend, decomp.seasonal, decomp.resid]

gp = plt.figure(figsize=(15,15)) 

    

gp.subplots_adjust(hspace=0.5)

for i in range(1, len(cols)+1):

    ax = gp.add_subplot(3,1, i)

    data[i-1].plot(ax=ax)

    ax.set_title('{}'.format(cols[i-1]))
plant1_sensor.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15,8), style='b.')

plant1_sensor.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)

plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')

plt.ylabel('Temperature(°C)')
module_temp = plant1_sensor.pivot_table(values='MODULE_TEMPERATURE', index='time', columns='date')
module_temp.boxplot(figsize=(15,5), grid=False, rot=90)

plt.title('MODULE TEMPERATURE BOXES')

plt.ylabel('Temperature (°C)')
multi_plot(module_temp, row=9,  col=4, title='Module Temp.')
#we can also see also calendar plot

mod_temp = plant1_sensor.groupby('date')['MODULE_TEMPERATURE'].agg('mean')
mod_temp.plot(grid=True, figsize=(15,5), legend=True)

plt.title('MODULE TEMPERATURE 15 MAY- 17 JUNE')

plt.ylabel('Temperature (°C)')
#we plot a %change of MODULE TEMPERATURE.

chan_mod_temp = (mod_temp.diff()/mod_temp)*100
chan_mod_temp.plot(grid=True, legend=True, figsize=(15,5))

plt.ylabel('%change')

plt.title('MODULE TEMPERATURE %change')
plant1_sensor.plot(x='time', y = 'IRRADIATION', style='.', legend=True, figsize=(15,5))

plant1_sensor.groupby('time')['IRRADIATION'].agg('mean').plot(legend=True, colormap='Reds_r')

plt.title('IRRADIATION')
irra = plant1_sensor.pivot_table(values='IRRADIATION', index='time', columns='date')
irra.tail()
irra.boxplot(figsize=(15,5), rot = 90, grid=False)

plt.title('IRRADIATION BOXES')
rad = plant1_sensor.groupby('date')['IRRADIATION'].agg('sum')
rad.plot(grid=True, figsize=(15,5), legend=True)

plt.title('IRRADIATION 15 MAY- 17 JUNE')
# we are merge our solar power generation data and weather sensor data

power_sensor = plant1_sensor.merge(plant1_data, left_on='DATE_TIME', right_on='DATE_TIME')
power_sensor.tail(3)
#we remove the columns that we do not need

del power_sensor['date_x']

del power_sensor['date_y']

del power_sensor['time_x']

del power_sensor['time_y']
power_sensor.tail(3)
power_sensor.info()
#we start correlation

power_sensor.corr(method = 'spearman')
corr = power_sensor.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']).corr(method = 'spearman')
plt.figure(dpi=100)

sns.heatmap(corr, robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)

plt.show()
# we make pairplot

sns.pairplot(power_sensor.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']))

plt.show()
#we plot dc power vs ac power
plt.figure(dpi=100)

sns.lmplot(x='DC_POWER', y='AC_POWER', data=power_sensor)

plt.title('Regression plot')

plt.show()
plt.figure(dpi=100)

sns.lmplot(x='AMBIENT_TEMPERATURE', y='DC_POWER', data=power_sensor)

plt.title('Regression plot')

plt.show()
plt.figure(dpi=100)

sns.lmplot(x='MODULE_TEMPERATURE', y='DC_POWER', data=power_sensor)

plt.title('Regression plot')

plt.show()
plt.figure(dpi=100)

sns.lmplot(x='IRRADIATION', y='DC_POWER', data=power_sensor)

plt.title('Regression plot')

plt.show()
# we introduce DELTA_TEMPERATURE

power_sensor['DELTA_TEMPERATURE'] = abs(power_sensor.AMBIENT_TEMPERATURE - power_sensor.MODULE_TEMPERATURE)
# we check if all is ok

power_sensor.tail(3)
#now we use correlation

power_sensor.corr(method='spearman')['DELTA_TEMPERATURE']
sns.lmplot(x='DELTA_TEMPERATURE', y='DC_POWER', data=power_sensor)

plt.title('correlation between DC_POWER and DELTA_TEMPERATURE')
sns.lmplot(x='DELTA_TEMPERATURE', y='IRRADIATION', data=power_sensor)

plt.title('Regression plot')
file2 = '/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv'
plant2_data = pd.read_csv(file2)
plant2_data.head(3)
plant2_data.info()
#we compute a sum of 22 inverters

plant2_data = plant2_data.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum').reset_index()
plant2_data['DATE_TIME'] = pd.to_datetime(plant2_data['DATE_TIME'], errors='coerce')

plant2_data['time'] = plant2_data['DATE_TIME'].dt.time

plant2_data['date'] = pd.to_datetime(plant2_data['DATE_TIME'].dt.date)
plant2_data.tail(3)
plant2_data.info()
#we conpare a dc power of two plant

ax = plant1_data.plot(x='time', y='DC_POWER', figsize=(15,5), legend=True, style='b.')

plant2_data.plot(x='time', y='DC_POWER', legend=True, style='r.', ax=ax)

plt.title('Plant1(blue) vs Plant2(red)')

plt.ylabel('Power (KW)')
#we conpare a dc power of two plant

ax1 = plant1_data.plot(x='time', y='AC_POWER', figsize=(15,5), legend=True, style='b.', )

plant2_data.plot(x='time', y='AC_POWER', legend=True, style='r.', ax=ax1)

plt.title('Plant1(blue) vs Plant2(red)')

plt.ylabel('Power (KW)')
p2_daily_dc = plant2_data.groupby('date')['DC_POWER'].agg('sum')
axh = daily_dc.plot.bar(legend=True, figsize=(15,5), color='Blue', label='DC_POWER Plant I')

p2_daily_dc.plot.bar(legend=True, color='Red', label='DC_POWER Plant II', stacked=False)

plt.title('DC POWER COMPARISON')

plt.ylabel('Power (KW)')

plt.show()
daily_ac = plant1_data.groupby('date')['AC_POWER'].agg('sum')

p2_daily_ac = plant2_data.groupby('date')['AC_POWER'].agg('sum')
ac = daily_ac.plot.bar(legend=True, figsize=(15,5), color='Blue', label='AC_POWER Plant I')

p2_daily_ac.plot.bar(legend=True, color='Red', label='AC_POWER Plant II')

plt.title('AC POWER COMPARISON')

plt.ylabel('Power (KW)')

plt.show()
#compute daily_yield for each date

p2_dyield = plant2_data.groupby('date')['DAILY_YIELD'].agg('sum')
dy = dyield.plot.bar(figsize=(15,5), legend=True, label='DAILY_YIELD PLANT I', color='Blue')

p2_dyield.plot.bar(legend=True, label='DAILY_YIELD PLANT II', color='Red')

plt.ylabel('Energy (KWh)')

plt.title('DAILY YIELD COMPARISON')
#compute a average total_yield for plant I for each day

tyield = plant1_data.groupby('date')['TOTAL_YIELD'].agg('mean')



#compute a average total_yield for plant II for each day

p2_tyield = plant2_data.groupby('date')['TOTAL_YIELD'].agg('mean')
aver = p2_tyield.plot.bar(figsize=(15,5), legend=True, label='AVERAGE TOTAL YIELD PLANT II', color='Red')

tyield.plot.bar(legend=True, label='AVERAGE TOTAL YIELD PLANT I', color='Blue',ax=aver)
file3 = '/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv'
plant2_sensor = pd.read_csv(file3)
plant2_sensor.tail()
plant2_sensor.info()
plant2_sensor['DATE_TIME'] = pd.to_datetime(plant2_sensor['DATE_TIME'], errors='coerce')
# same work cleaning data for plant II

plant2_sensor['date'] = pd.to_datetime(pd.to_datetime(plant2_sensor['DATE_TIME']).dt.date)

plant2_sensor['time'] = pd.to_datetime(plant2_sensor['DATE_TIME']).dt.time





del plant2_sensor['PLANT_ID']

del plant2_sensor['SOURCE_KEY']
plant2_sensor.head()
plant1_sensor[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','time']].plot(x='time', label='Plant I', title='PLANT I', figsize=(15,5), style='.')

plant2_sensor[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','time']].plot(x='time', label='Plant II', title='PLANT II', figsize=(15,5), style='.')

plt.ylabel('Temperature (°C)')
#compare IRRADIATION PLANT I VS PLANT II

aq = plant1_sensor.plot(x='time', y='IRRADIATION', legend=True, label='IRRADIATION PLANT I', color='Blue', style='.', figsize=(15,5))

plant2_sensor.plot(x='time', y='IRRADIATION', legend=True, label='IRRADIATION PLANT II',  color='Red', style='.', ax=aq)

plt.title('IRRADIATION COMPARISON')
# we are merging our solar power generation data and weather sensor data for plant 2

sensorData = plant2_sensor.merge(plant2_data, left_on='DATE_TIME', right_on='DATE_TIME')
#we remove the columns that we do not need

del sensorData['date_x']

del sensorData['date_y']

del sensorData['time_x']

del sensorData['time_y']
sensorData.tail()
sensorData = sensorData.assign(DELTA_TEMPERATURE = abs(sensorData.MODULE_TEMPERATURE - sensorData.AMBIENT_TEMPERATURE),

                              NEW_DAILY_YIELD = sensorData.DAILY_YIELD.diff(),

                              NEW_TOTAL_YIELD = sensorData.TOTAL_YIELD.diff(),

                              NEW_AMBIENT_TEMPERATURE = sensorData.AMBIENT_TEMPERATURE.diff(),

                              NEW_MODULE_TEMPERATURE = sensorData.MODULE_TEMPERATURE.diff(),

                              NEW_AC_POWER = sensorData.AC_POWER.diff())
#see

sensorData.head()
sensorData.corr(method='spearman').style.background_gradient('viridis')
plt.figure(dpi=100, figsize=(15,10))

sns.heatmap(sensorData.corr(method='spearman'), robust=True, annot=True, fmt='0.2f', linewidths=.5, square=False)

plt.show()
#we plot ac vs dc power

sns.lmplot(x='DC_POWER', y='AC_POWER', data=sensorData)

plt.title('Regression plot')
#we plot New DAILY YIELD vs ac power

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(x='AC_POWER', y='NEW_DAILY_YIELD', data=sensorData)

plt.title('Regression plot')
#we plot New DAILY YIELD vs IRRADIATION

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(x='IRRADIATION', y='NEW_DAILY_YIELD', data=sensorData)

plt.title('Regression plot')
#we plot New DAILY YIELD vs ac power

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(x='MODULE_TEMPERATURE', y='NEW_DAILY_YIELD', data=sensorData)

plt.title('Regression plot')
#we plot New DAILY YIELD vs DELTA TEMPERATURE

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(x='DELTA_TEMPERATURE', y='NEW_DAILY_YIELD', data=sensorData)

plt.title('Regression plot')
#we plot New TOTAL YIELD vs New daily yield

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(y='NEW_TOTAL_YIELD', x='NEW_DAILY_YIELD', data=sensorData)

plt.title('Regression plot')
#we plot New TOTAL YIELD vs New daily yield

plt.figure(dpi=(100), figsize=(15,5))

sns.regplot(y='NEW_AC_POWER', x='NEW_MODULE_TEMPERATURE', data=sensorData)

plt.title('Regression plot')