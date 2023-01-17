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
#load data from csv files

df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_pwea1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_pwea2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
# pre-processing date and time

df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df_pgen1['DATE']= df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME']= df_pgen1['DATE_TIME'].apply(lambda x:x.time())

df_pgen1['DATE']= pd.to_datetime(df_pgen1['DATE'],format ='%Y-%m-%d')

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute





df_pwea1['DATE_TIME'] = pd.to_datetime(df_pwea1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

df_pwea1['DATE']= df_pwea1['DATE_TIME'].apply(lambda x:x.date())

df_pwea1['TIME']= df_pwea1['DATE_TIME'].apply(lambda x:x.time())

df_pwea1['DATE']= pd.to_datetime(df_pwea1['DATE'],format ='%Y-%m-%d')

df_pwea1['HOUR'] = pd.to_datetime(df_pwea1['TIME'],format='%H:%M:%S').dt.hour

df_pwea1['MINUTES'] = pd.to_datetime(df_pwea1['TIME'],format='%H:%M:%S').dt.minute



df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df_pgen2['DATE']= df_pgen2['DATE_TIME'].apply(lambda x:x.date())

df_pgen2['TIME']= df_pgen2['DATE_TIME'].apply(lambda x:x.time())

df_pgen2['DATE']= pd.to_datetime(df_pgen2['DATE'],format ='%Y-%m-%d')

df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour 

df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute



df_pwea2['DATE_TIME'] = pd.to_datetime(df_pwea2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

df_pwea2['DATE']= df_pwea2['DATE_TIME'].apply(lambda x:x.date())

df_pwea2['TIME']= df_pwea2['DATE_TIME'].apply(lambda x:x.time())

df_pwea2['DATE']= pd.to_datetime(df_pwea2['DATE'],format ='%Y-%m-%d')

df_pwea2['HOUR'] = pd.to_datetime(df_pwea2['TIME'],format='%H:%M:%S').dt.hour

df_pwea2['MINUTES'] = pd.to_datetime(df_pwea2['TIME'],format='%H:%M:%S').dt.minute





#Exploration

df_pgen1['AC_POWER'].mean()
df_pgen1.head(15)
df_pgen1.tail(5)
df_pgen1.describe()
df_pgen1.value_counts()
#mean of daily yield

df_pgen1['DAILY_YIELD'].mean()

print('mean of plant 1',df_pgen1['DAILY_YIELD'].mean())

df_pgen2['DAILY_YIELD'].mean()

print('mean of plant 2',df_pgen2['DAILY_YIELD'].mean())



sum = (df_pgen1['DAILY_YIELD'].mean() + df_pgen2['DAILY_YIELD'].mean() )/2

print ('mean of daily yield for both together',sum)

#total irradiation per day

df_pwea1['IRRADIATION'].sum()
df_pwea2['IRRADIATION'].sum()
#ambient and module temp



print ('mat of plant1', df_pwea1['AMBIENT_TEMPERATURE'].max())

print ('mmt of plant1', df_pwea1['MODULE_TEMPERATURE'].max())

print ('mat of plant2', df_pwea2['AMBIENT_TEMPERATURE'].max())

print ('mmt of plant2', df_pwea2['MODULE_TEMPERATURE'].max())
# number of inverters

print('in plant1',len(df_pgen1['SOURCE_KEY'].unique()))

print('in plant2',len(df_pgen1['SOURCE_KEY'].unique()))
#What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?

df_pgen1['DC_POWER'].max()
gen1 = df_pgen1.groupby(['DATE']).sum()

gen1
df_pgen1['DC_POWER'].min()
gen2 = df_pgen2.groupby(['DATE']).sum()

gen2
dcmax_plant1 = (gen1['DC_POWER'].max())

dcmax_plant2 = (gen2['DC_POWER'].max())

dcmin_plant1 = (gen1['DC_POWER'].min())

dcmin_plant2 = (gen2['DC_POWER'].min())
print(dcmax_plant1)

print(dcmax_plant2)

print(dcmin_plant1)

print(dcmin_plant2)
acmax_plant1 = (gen1['AC_POWER'].max())

acmax_plant2 = (gen2['AC_POWER'].max())

acmin_plant1 = (gen1['AC_POWER'].min())

acmin_plant2 = (gen2['AC_POWER'].min())
print(acmax_plant1)

print(acmax_plant2)

print(acmin_plant1)

print(acmin_plant2)
# inverter that produces max ac power

dff = df_pgen1.groupby(['AC_POWER']).max()

dff

dff1 = dff.loc[1410.950000]

dff1
dff1.iloc[2]
dgg = df_pgen2.groupby(['AC_POWER']).max()

dgg
dgg1 = dgg.loc[1385.420000]

dgg1
dgg1.iloc[2]
#ranking inverters based on power produced

dhh = df_pgen1.groupby(['SOURCE_KEY']).count()

dhh
dhh1 = dhh.iloc[:,[2,3]].rank()

dhh1
dii = df_pgen2.groupby('SOURCE_KEY').count()

dii
dii1 = dii.iloc[:,[2,3]].rank()

dii1
#check for missing values

df_pgen1.isnull()
df_pwea1.isnull()
df_pgen2.isnull()
df_pwea2.isnull()