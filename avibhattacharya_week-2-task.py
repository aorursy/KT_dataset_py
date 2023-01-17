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
import pandas as pd
import matplotlib.pyplot as plt 
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M')   #datetime 
df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date   #split
df1['DATE'] = pd.to_datetime(df1['DATE'] )      #datetime series
df1.info()

df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')   #datetime 
df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date   #split
df2['DATE'] = pd.to_datetime(df2['DATE'] )      #datetime series
df2.info()
plt.figure(figsize=(20,10))
plt.plot(df1['DATE_TIME'],df1['DC_POWER'],label ='DC power')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('DC_POWER')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("DC_POWER VS Date_time")
plt.show()
plt.figure(figsize=(20,10))
plt.plot(df1['DATE_TIME'],df1['AC_POWER'],label ='AC power')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('AC_POWER')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("AC_POWER VS Date_time")
plt.show()
plt.figure(figsize=(20,10))
plt.plot(df1['DATE_TIME'],df1['DAILY_YIELD'],label ='Yield')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('DAILY_YIELD')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("DAILY_YIELD VS Date_time")
plt.show()
plt.figure(figsize=(20,10))
plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE'],label ='Module Temperature')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('MODULE TEMPERATURE')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("Module VS Date_time")
plt.show()

plt.figure(figsize=(20,10))
plt.plot(df2['DATE_TIME'],df2['AMBIENT_TEMPERATURE'],label ='Ambient Temperature')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('AMBIENT TEMPERATURE')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("Ambient VS Date_time")
plt.show()
plt.figure(figsize=(20,10))
plt.plot(df2['DATE_TIME'],df2['IRRADIATION'],label ='Irradiation')
plt.legend()
plt.xlabel('DATE TIME')
plt.ylabel('IRRADIATION')
plt.xticks(rotation = 90)
plt.grid()
plt.margins(0.05)
plt.title("IRRADIATION VS Date_time")
plt.show()
plt.figure(figsize=(18,9))
plt.plot(df2['AMBIENT_TEMPERATURE'],df2['MODULE_TEMPERATURE'],marker = 'o' ,linestyle = '', alpha = 0.5, ms=10)
plt.xlabel('AMBIENT_TEMPERATURE')
plt.ylabel('MODULE_TEMPERATURE')
plt.title("AMBIENT_TEMPERATURE VS MODULE_TEMPERATURE")
plt.figure(figsize=(18,9))
plt.plot(df1['DC_POWER'],df1['AC_POWER'],marker = 'o' ,linestyle = '', alpha = 0.5, ms=10)
plt.xlabel('DC_POWER')
plt.ylabel('AC_POWER')
plt.title(" AC_POWER VS DC_POWER")
plt.figure(figsize=(18,9))
plt.plot(df2['AMBIENT_TEMPERATURE'],df2['IRRADIATION'],marker = 'o' ,linestyle = '', alpha = 0.5, ms=10)
plt.xlabel('AMBIENT_TEMPERATURE')
plt.ylabel('IRRADIATION')
plt.title(" IRRADIATION VS AMBIENT_TEMPERATURE")
r_left = pd.merge(df1,df2, on ='DATE_TIME', how='left')
r_left
plt.figure(figsize=(18,9))
plt.plot(r_left['IRRADIATION'], r_left['DC_POWER'], marker= 'o', linestyle= '')
plt.xlabel('IRRADIATION')
plt.ylabel('DC_POWER')
plt.show()
plt.figure(figsize=(18,9))
plt.plot(r_left['MODULE_TEMPERATURE'], r_left['DC_POWER'], marker= 'o', label= 'DC_POWER', linestyle= '')
plt.xlabel('MODULE_TEMPERATURE')
plt.ylabel('DC_POWER')
plt.legend()
plt.show()
dates = r_left['DATE_x'].unique()
dates

plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'], marker = 'o', linestyle ='', alpha = 0.5, label = pd.to_datetime(date,format = '%Y-%m-%d').date())
plt.xlabel('AMBIENT_TEMPERATURE')
plt.ylabel('MODULE_TEMPERATURE')
plt.legend()
plt.show()
data = r_left[r_left['DATE_x']==dates[0]][r_left['IRRADIATION' ]>0.1]
plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'], marker='o',linestyle='',label = pd.to_datetime(dates[0],format = '%Y-%m-%d').date())
plt.xlabel('MODULE_TEMPERATURE')
plt.ylabel('DC_POWER')
plt.legend()
plt.show()

plt.figure(figsize=(19,9))
for date in dates:
    data = df2[df2['DATE']==date][df2['IRRADIATION']>0]
    plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'], marker = 'o', linestyle ='', alpha = 0.5, ms= 6,label = pd.to_datetime(date,format = '%Y-%m-%d').date())

plt.legend()
plt.show()
plt.figure(figsize=(20,10))
for date in dates:
    data = r_left[r_left['DATE_x']==date][r_left['IRRADIATION']>0.1]
    plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'],marker='o',linestyle='',label = pd.to_datetime(date,format='%Y-%m-%d').date())
plt.legend()
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.title('DC POWER VS MODULE TEMPERATURE')
plt.show()
