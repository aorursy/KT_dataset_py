# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pgd1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
wsd1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
pgd2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
wsd2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
pgd1['DATE_TIME']=pd.to_datetime(pgd1['DATE_TIME'],format= '%d-%m-%Y %H:%M')
pgd1['DATE']=pgd1['DATE_TIME'].apply(lambda x:x.date())
pgd1['TIME']=pgd1['DATE_TIME'].apply(lambda x:x.time())
wsd1['DATE_TIME']=pd.to_datetime(wsd1['DATE_TIME'],format= '%Y-%m-%d %H:%M:%S')
wsd1['DATE']=wsd1['DATE_TIME'].apply(lambda x:x.date())
wsd1['TIME']=wsd1['DATE_TIME'].apply(lambda x:x.time())
pgd2['DATE_TIME']=pd.to_datetime(pgd2['DATE_TIME'],format= '%Y-%m-%d %H:%M')
pgd2['DATE']=pgd2['DATE_TIME'].apply(lambda x:x.date())
pgd2['TIME']=pgd2['DATE_TIME'].apply(lambda x:x.time())
wsd2['DATE_TIME']=pd.to_datetime(wsd2['DATE_TIME'],format= '%Y-%m-%d %H:%M:%S')
wsd2['DATE']=wsd2['DATE_TIME'].apply(lambda x:x.date())
wsd2['TIME']=wsd2['DATE_TIME'].apply(lambda x:x.time())

pgd1['DATE'] = pd.to_datetime(pgd1['DATE'],format = '%Y-%m-%d')
pgd2['DATE'] = pd.to_datetime(pgd2['DATE'],format = '%Y-%m-%d')
wsd1['DATE'] = pd.to_datetime(wsd1['DATE'],format = '%Y-%m-%d')
wsd2['DATE'] = pd.to_datetime(wsd2['DATE'],format = '%Y-%m-%d')

pgd1['TIME'] = pd.to_datetime(pgd1['TIME'],format = '%H:%M:%S')
pgd2['TIME'] = pd.to_datetime(pgd2['TIME'],format = '%H:%M:%S')
wsd1['TIME'] = pd.to_datetime(wsd1['TIME'],format = '%H:%M:%S')
wsd2['TIME'] = pd.to_datetime(wsd2['TIME'],format = '%H:%M:%S')

pgd1['HOUR'] = pd.to_datetime(pgd1['TIME'],format='%H:%M:%S').dt.hour
wsd1['HOUR'] = pd.to_datetime(wsd1['TIME'],format='%H:%M:%S').dt.hour
pgd2['HOUR'] = pd.to_datetime(pgd2['TIME'],format='%H:%M:%S').dt.hour
wsd2['HOUR'] = pd.to_datetime(wsd2['TIME'],format='%H:%M:%S').dt.hour
rleft1 = pd.merge(pgd1,wsd1,on='DATE_TIME',how='left')
rleft1['DATE']=rleft1['DATE_TIME'].apply(lambda x:x.date())
rleft1['TIME']=rleft1['DATE_TIME'].apply(lambda x:x.time())
rleft2 = pd.merge(pgd2,wsd2,on='DATE_TIME',how='left')
rleft1
import matplotlib.pyplot as plt
import seaborn as sns
import fbprophet
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import itertools
from statsmodels.tsa.arima_model import ARIMA
#graph of  daily yield vs datetime for pgd1
sns.set_style('whitegrid');
sns.FacetGrid(pgd1,hue='SOURCE_KEY',size=11)\
   .map(plt.scatter,'DATE_TIME','DAILY_YIELD')\
   .add_legend();
plt.show();
fig = plt.figure(figsize =(10, 9)) 
pgd1.groupby(pgd1['SOURCE_KEY'])['DAILY_YIELD'].max().plot.bar()
plt.grid()
plt.title('DAILY YIELD of each INVERTOR')
plt.ylabel('daily YIELD')
plt.show()
fig = plt.figure(figsize =(10, 9)) 
pgd1.groupby(pgd1['SOURCE_KEY'])['TOTAL_YIELD'].max().plot.bar()
plt.grid()
plt.title('TOTAL YIELD of each INVERTOR')
plt.ylabel('TOTAL YIELD')
plt.show()
plt.figure(figsize=(15,8))
plt.plot(pgd1['DATE_TIME'],
        pgd1['DAILY_YIELD'])

plt.title('Correlation Between  DAILY_YIELD& DATE_TIME')
plt.xlabel('DATE_TIME')
plt.ylabel('DAILY_YIELD')
plt.show()
inv_summary = pgd1.groupby(['SOURCE_KEY','DATE']).agg(DAILY_YIELD = ('DAILY_YIELD',max),INV = ('SOURCE_KEY', max))
import seaborn as sns
sns.set(style="ticks")

f, ax = plt.subplots(figsize=(10, 12))

sns.boxplot(x="DAILY_YIELD", y="INV", data=inv_summary,
            whis=[0, 100], palette="vlag")


# Tweak the visual presentation
#ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.grid()
ax.margins(0.01)
ax.set(ylabel="Inverters")
sns.despine(trim=True, left=True)
print('The inverter that has the highest daily yield for pgd1 is '+ str(pgd1.iloc[pgd1['DAILY_YIELD'].argmax()]['SOURCE_KEY']))
print('The inverter that has the lowest daily yield for pgd1 is '+ str(pgd1.iloc[pgd1['DAILY_YIELD'].argmin()]['SOURCE_KEY']))
print('The inverter that has the highest daily yield for pgd2 is '+ str(pgd2.iloc[pgd2['DAILY_YIELD'].argmax()]['SOURCE_KEY']))
print('The inverter that has the lowest  yield for pgd2 is '+ str(pgd2.iloc[pgd2['DAILY_YIELD'].argmin()]['SOURCE_KEY']))
print('---------------------------------------------------------------------------------------')
print('The inverter that has the highest total yield for pgd1 is '+ str(pgd1.iloc[pgd1['TOTAL_YIELD'].argmax()]['SOURCE_KEY']))
print('The inverter that has the lowest total yield for pgd1 is '+ str(pgd1.iloc[pgd1['TOTAL_YIELD'].argmin()]['SOURCE_KEY']))
print('The inverter that has the highest total yield for pgd2 is '+ str(pgd2.iloc[pgd2['TOTAL_YIELD'].argmax()]['SOURCE_KEY']))
print('The inverter that has the lowest total yield for pgd2 is '+ str(pgd2.iloc[pgd2['TOTAL_YIELD'].argmin()]['SOURCE_KEY']))
print('The highest daily yield for pgd1 is '+ str(pgd1['DAILY_YIELD'].max()))
print('The lowest daily yield for pgd1 is '+ str(pgd1['DAILY_YIELD'].min()))
print('The highest daily yield for pgd2 is '+ str(pgd2['DAILY_YIELD'].max()))
print('The lowest daily yield for pgd2 is '+ str(pgd2['DAILY_YIELD'].min()))
print('-------------------------------------------------------------------')
print('The highest total yield for pgd1 is '+ str(pgd1['TOTAL_YIELD'].max()))
print('The lowest total yield for pgd1 is '+ str(pgd1['TOTAL_YIELD'].min()))
print('The highest total yield for pgd2 is '+ str(pgd2['TOTAL_YIELD'].max()))
print('The lowest total yield for pgd2 is '+ str(pgd2['TOTAL_YIELD'].min()))

print('For pgd1 daily yield was highest at the datetime-'+str(pgd1.iloc[pgd1['DAILY_YIELD'].argmax()]['DATE_TIME']))
print('For pgd1 daily yield was lowest at the datetime-'+str(pgd1.iloc[pgd1['DAILY_YIELD'].argmin()]['DATE_TIME']))
print('For pgd2 daily yield was highest at the datetime-'+str(pgd2.iloc[pgd2['DAILY_YIELD'].argmax()]['DATE_TIME']))
print('For pgd2 daily yield was lowest at the datetime-'+str(pgd2.iloc[pgd2['DAILY_YIELD'].argmin()]['DATE_TIME']))
print('-------------------------------------------------------------------------------------')
print('For pgd1 Total yield was highest at the datetime-'+str(pgd1.iloc[pgd1['TOTAL_YIELD'].argmax()]['DATE_TIME']))
print('For pgd1 Total yield was lowest at the datetime-'+str(pgd1.iloc[pgd1['TOTAL_YIELD'].argmin()]['DATE_TIME']))
print('For pgd2 Total yield was highest at the datetime-'+str(pgd2.iloc[pgd2['TOTAL_YIELD'].argmax()]['DATE_TIME']))
print('For pgd2 Total yield was lowest at the datetime-'+str(pgd2.iloc[pgd2['TOTAL_YIELD'].argmin()]['DATE_TIME']))
print('For pgd1 dc power was highest at the datetime- '+str(pgd1.iloc[pgd1['DC_POWER'].argmax()]['DATE_TIME']))
print('For pgd1 ac power was highest at the datetime- '+str(pgd1.iloc[pgd1['AC_POWER'].argmax()]['DATE_TIME']))
print('For pgd2 dc power was highest at the datetime- '+str(pgd2.iloc[pgd1['DC_POWER'].argmax()]['DATE_TIME']))
print('For pgd2 ac power was highest at the datetime- '+str(pgd2.iloc[pgd1['AC_POWER'].argmax()]['DATE_TIME']))
print('------------------------------------------------------------------------------------------')
print('For pgd1 the ac power was lowest at the date- '+str(pgd1.iloc[pgd1['AC_POWER'].argmin()]['DATE_TIME']))
print('For pgd1 the dc power was lowest at the date- '+str(pgd1.iloc[pgd1['DC_POWER'].argmin()]['DATE_TIME']))
print('For pgd2 the ac power was lowest at the date- '+str(pgd2.iloc[pgd2['AC_POWER'].argmin()]['DATE_TIME']))
print('For pgd2 the dc power was lowest at the date- '+str(pgd2.iloc[pgd2['DC_POWER'].argmin()]['DATE_TIME']))
#total yield for the lowest acpower date
print(pgd1[pgd1['DATE']=='2020-05-15']['TOTAL_YIELD'].max())
print(pgd2[pgd2['DATE']=='2020-06-14']['TOTAL_YIELD'].max())
#yield for the date in which there has been highest acpower
print(pgd1[pgd1['DATE']=='2020-06-14']['DAILY_YIELD'].max())
print(pgd2[pgd2['DATE']=='2020-06-15']['DAILY_YIELD'].max())
#yield for thedatae in which ac power is highest
print(pgd1[pgd1['DATE']=='2020-06-14']['TOTAL_YIELD'].max())

print(pgd2[pgd2['DATE']=='2020-06-15']['TOTAL_YIELD'].max())
#date at which daily yield is high
pgd1[pgd1['DATE']=='2020-05-25']['SOURCE_KEY'].value_counts()
#date at which acpoower is highest
pgd1[pgd1['DATE']=='2020-06-14']['SOURCE_KEY'].value_counts()

#there must be 22*4=88 datas

pgd1[pgd1['DATE']=='2020-06-14']['HOUR'].value_counts()
#daily yield highest date
pgd1[pgd1['DATE']=='2020-05-25']['HOUR'].value_counts()
sns.set_style('whitegrid');
sns.FacetGrid(pgd1,hue='DATE',size=11)\
   .map(plt.scatter,'AC_POWER','TOTAL_YIELD')\
   .add_legend();
plt.show();
p=pgd1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max().reset_index()
print('The maximum total_yield for each inverter in pgd1 is:')
print(p)
i=pgd1.groupby('SOURCE_KEY')['TOTAL_YIELD'].min().reset_index()
print('The minimum total_yield for each inverter in pgd1 is:')
print(i)
q=pgd2.groupby('SOURCE_KEY')['TOTAL_YIELD'].max().reset_index()
print('The maximum total_yield for each inverter in pgd2 is:')
print(q)
j=pgd2.groupby('SOURCE_KEY')['TOTAL_YIELD'].min().reset_index()
print('The minimum total_yield for each inverter in pgd2 is:')
print(j)
k=pgd2.groupby('DATE').agg(TYIELD= ('TOTAL_YIELD', min))
k
                                         
l=pgd2.groupby('SOURCE_KEY').agg(DAILY_YIELD= ('DAILY_YIELD', max))
l
j=pgd2.groupby('SOURCE_KEY')['TOTAL_YIELD'].min().reset_index()
print('The minimum total_yield for each inverter in pgd2 is:')
print(j)
tempo = pgd2.groupby(['DATE_TIME']).agg(TOTAL_YIELD = ('TOTAL_YIELD',min))

tempo
pgd2[pgd2['DATE_TIME']=='2020-06-08 05:00:00']
ds1 = pgd1.groupby('DATE').agg(TYIELD= ('TOTAL_YIELD', max),
                                         DATE = ('DATE',max))
print(ds1)
                              
ds1 = ds1.rename(columns={'DATE': 'ds', 'TYIELD': 'y'})
print(ds1)
#total yield prediction  for the next 30 days using fb prophet for pgd1
prophet_pgd1 = fbprophet.Prophet(changepoint_prior_scale=0.25) 

prophet_pgd1.fit(ds1)

forecast_pgd1 = prophet_pgd1.make_future_dataframe(periods=90, freq='D')

forecast_pgd1 = prophet_pgd1.predict(forecast_pgd1)

prophet_pgd1.plot(forecast_pgd1, xlabel = 'Date', ylabel = 'total_yield')
plt.title('Total yield Prediction')
#total yield prediction  for the next 30 days using fb prophet for pgd2
prophet_pgd2 = fbprophet.Prophet(changepoint_prior_scale=0.25) 

prophet_pgd2.fit(ds1)

forecast_pgd2 = prophet_pgd2.make_future_dataframe(periods=90, freq='D')

forecast_pgd2 = prophet_pgd2.predict(forecast_pgd2)

prophet_pgd2.plot(forecast_pgd2, xlabel = 'Date', ylabel = 'total_yield')
plt.title('Total yield Prediction')


