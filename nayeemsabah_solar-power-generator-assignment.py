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
df_pgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wsen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wsen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1
df_wsen1
df_pgen2
df_wsen2
df_pgen1.describe()
df_wsen1.describe()
df_pgen2.describe()
df_wsen2.describe()
df_pgen1.info()
df_wsen1.info()
df_pgen2.info()
df_wsen2.info()
dy1=df_pgen1['DAILY_YIELD'].mean()
print("The mean daily yield of plant 1 is", dy1)
dy2=df_pgen2['DAILY_YIELD'].mean()
print("The mean daily yield of plant 2 is", dy2)
#Converting DATE_TIME column from object to datetime
df_wsen1['DATE_TIME']=pd.to_datetime(df_wsen1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
df_wsen2['DATE_TIME']=pd.to_datetime(df_wsen2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
#Creating separate columns for DATE and TIME
df_wsen1['DATE'] = df_wsen1['DATE_TIME'].apply(lambda x:x.date())
df_wsen1['TIME'] = df_wsen1['DATE_TIME'].apply(lambda x:x.time())
df_wsen2['DATE'] = df_wsen2['DATE_TIME'].apply(lambda x:x.date())
df_wsen2['TIME'] = df_wsen2['DATE_TIME'].apply(lambda x:x.time())
#Converting DATE column to date
df_wsen1['DATE'] = pd.to_datetime(df_wsen1['DATE'],format = '%Y-%m-%d')
df_wsen2['DATE'] = pd.to_datetime(df_wsen2['DATE'],format = '%Y-%m-%d')

#Converting TIME column to hours and minutes
df_wsen1['HOUR'] = pd.to_datetime(df_wsen1['TIME'],format='%H:%M:%S').dt.hour
df_wsen1['MINUTES'] = pd.to_datetime(df_wsen1['TIME'],format='%H:%M:%S').dt.minute
df_wsen2['HOUR'] = pd.to_datetime(df_wsen2['TIME'],format='%H:%M:%S').dt.hour
df_wsen2['MINUTES'] = pd.to_datetime(df_wsen2['TIME'],format='%H:%M:%S').dt.minute
df_wsen1.info()
df_wsen2.info()
print("The total irradiation per day for Plant 1 is as follows: ")
df_wsen1.groupby(df_wsen1['DATE'])["IRRADIATION"].sum().reset_index()

print("The total irraadiation per day for Plant 2 is as follows: ")
df_wsen2.groupby(df_wsen2['DATE'])["IRRADIATION"].sum().reset_index()
mat1=df_wsen1['AMBIENT_TEMPERATURE'].max()
mmt1=df_wsen1['MODULE_TEMPERATURE'].max()
print("The maximum ambient temperature of plant 1 is", mat1)
print("The maximum module temperature of plant 1 is", mmt1)
mat2=df_wsen2['AMBIENT_TEMPERATURE'].max()
mmt2=df_wsen2['MODULE_TEMPERATURE'].max()
print("The maximum ambient temperature of plant 2 is", mat2)
print("The maximum module temperature of plant 2 is", mmt2)
ni1=len(df_pgen1['SOURCE_KEY'].unique())
print("There are", ni1,"inverters in Plant 1")
ni2=len(df_pgen2['SOURCE_KEY'].unique())
print("There are", ni2,"inverters in Plant 2")
#Converting DATE_TIME column from object to datetime
df_pgen1['DATE_TIME']=pd.to_datetime(df_pgen1['DATE_TIME'],format='%d-%m-%Y %H:%M')
df_pgen2['DATE_TIME']=pd.to_datetime(df_pgen2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
#Creating separate columns for DATE and TIME
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
#Converting DATE column to date
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')

#Converting TIME column to hours and minutes
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.info()
df_pgen2.info()
print("The max DC Power produced in Plant 1 is: ")
df_pgen1.groupby(df_pgen1['DATE_TIME'])['DC_POWER'].max().reset_index()
print("THe min DC Power produced in Plant 1 is: ")
df_pgen1.groupby(df_pgen1['DATE_TIME'])['DC_POWER'].min().reset_index()
print("The max AC Power produced in plant 1 is: ")
df_pgen1.groupby(df_pgen1['DATE'])['AC_POWER'].max().reset_index()
print("The min AC Power prodeced in Plant 1 is: ")
df_pgen1.groupby(df_pgen1['DATE'])['AC_POWER'].min().reset_index()
print("The max DC Power prodeuced in Plant 2 is:")
df_pgen2.groupby(df_pgen2['DATE'])['DC_POWER'].max().reset_index()
print("The min DC Power prodeuced in Plant 2 is:")
df_pgen2.groupby(df_pgen2['DATE'])['DC_POWER'].min().reset_index()
print("THe max AC Power produced in Plant 2 is: ")
df_pgen2.groupby(df_pgen2['DATE_TIME'])['AC_POWER'].max().reset_index()
print("THe min AC Power produced in Plant 2 is: ")
df_pgen2.groupby(df_pgen2['DATE_TIME'])['AC_POWER'].min().reset_index()
dp1=df_pgen1.iloc[df_pgen1['DC_POWER'].argmax()]["SOURCE_KEY"]
print("The inverter with source code", dp1,"has produced the maximum AC Power in Plant 1")
ap1=df_pgen1.iloc[df_pgen1['AC_POWER'].argmax()]["SOURCE_KEY"]
print("The inverter with source code", ap1,"has produced the maximum DC Power in Plant 1")
dp2=df_pgen2.iloc[df_pgen2['DC_POWER'].argmax()]["SOURCE_KEY"]
print("The inverter with source code", dp2,"has produced the maximum AC Power in Plant 2")
ap2=df_pgen2.iloc[df_pgen2['AC_POWER'].argmax()]["SOURCE_KEY"]
print("The inverter with source code", ap2,"has produced the maximum DC Power in Plant 2")
df_pgen1.sort_values(by='DC_POWER',ascending=False).reset_index()
df_pgen1.sort_values(by='AC_POWER',ascending=False).reset_index()
pgd1['SOURCE_KEY'].value_counts()
34*24*4
wsd1['SOURCE_KEY'].value_counts()
pgd2['SOURCE_KEY'].value_counts()
wsd2['SOURCE_KEY'].value_counts()