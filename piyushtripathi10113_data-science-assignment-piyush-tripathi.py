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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_pgen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
df_sen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_pgen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
df_sen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_sen1['DATE_TIME'] = pd.to_datetime(df_sen1['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_sen1['DATE'] = df_sen1['DATE_TIME'].apply(lambda x:x.date())
df_sen1['TIME'] = df_sen1['DATE_TIME'].apply(lambda x:x.time())
df_sen1['DATE'] = pd.to_datetime(df_sen1['DATE'],format = '%Y-%m-%d')
df_sen1['HOUR'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.hour
df_sen1['MINUTES'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute
df_sen2['DATE_TIME'] = pd.to_datetime(df_sen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_sen2['DATE'] = df_sen1['DATE_TIME'].apply(lambda x:x.date())
df_sen2['TIME'] = df_sen1['DATE_TIME'].apply(lambda x:x.time())
df_sen2['DATE'] = pd.to_datetime(df_sen1['DATE'],format = '%Y-%m-%d')
df_sen2['HOUR'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.hour
df_sen2['MINUTES'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.minute
print("The Mean of 'DAILY_YIELD' over the dataset 'Plant_1_Generation_Data' is",df_pgen1['DAILY_YIELD'].mean())
df_pgen1.plot('DATE','DAILY_YIELD', kind = 'line');
print("The Mean of 'DAILY_YIELD' over the dataset 'Plant_2_Generation_Data' is",df_pgen2['DAILY_YIELD'].mean())
df_pgen2.plot('DATE','DAILY_YIELD', kind = 'line');
print("Total IRRADIATION per day is :")
print(df_sen1.groupby(['DATE']).sum()['IRRADIATION'])
df_sen1.groupby(['SOURCE_KEY']).plot('HOUR','IRRADIATION', kind = 'box');
print("Total IRRADIATION per day is :")
print(df_sen2.groupby(['DATE']).sum()['IRRADIATION'])
df_sen2.groupby(['SOURCE_KEY']).plot('HOUR','IRRADIATION', kind = 'box');
print('Max. Ambient Temperature is :',df_sen1['AMBIENT_TEMPERATURE'].max())
print('Max. Module Temperature is :',df_sen1['MODULE_TEMPERATURE'].max())
df_sen1.groupby(['SOURCE_KEY']).plot('AMBIENT_TEMPERATURE','MODULE_TEMPERATURE', kind = 'scatter');
print('Max. Ambient Temperature is :',df_sen2['AMBIENT_TEMPERATURE'].max())
print('Max. Module Temperature is :',df_sen2['MODULE_TEMPERATURE'].max())
df_sen2.groupby(['SOURCE_KEY']).plot('AMBIENT_TEMPERATURE','MODULE_TEMPERATURE', kind = 'scatter');
print("The number of Inverters in Plant 1 are :",len(df_pgen1['SOURCE_KEY'].unique()))
print("The number of Inverters in Plant 2 are :",len(df_pgen2['SOURCE_KEY'].unique()))
df_pgen1.groupby('DATE').max('DC_POWER')['DC_POWER']
df_pgen1.groupby('DATE').min('DC_POWER')['DC_POWER']
df_pgen2.groupby('DATE').max('DC_POWER')['DC_POWER']
df_pgen1.groupby('DATE').min('DC_POWER')['DC_POWER']
df_pgen1.groupby('DATE').max('AC_POWER')['AC_POWER']
df_pgen1.groupby('DATE').min('AC_POWER')['AC_POWER']
df_pgen2.groupby('DATE').max('AC_POWER')['AC_POWER']
df_pgen2.groupby('DATE').min('AC_POWER')['AC_POWER']
df_pgen1.groupby("SOURCE_KEY").max('DC_POWER').max()
df=df_pgen1.groupby("SOURCE_KEY").max('DC_POWER')['DC_POWER']
df.plot()
plt.xticks(rotation=45)
df_pgen2.groupby("SOURCE_KEY").max('DC_POWER').max()
df=df_pgen2.groupby("SOURCE_KEY").max('DC_POWER')['DC_POWER']
df.plot()
plt.xticks(rotation=45)
df_pgen1.groupby('SOURCE_KEY').max().sort_values(by=['DC_POWER'],ascending=False)['DC_POWER']
df_pgen1.groupby('SOURCE_KEY').max().sort_values(by=['AC_POWER'],ascending=False)['AC_POWER']
df_pgen2.groupby('SOURCE_KEY').max().sort_values(by=['DC_POWER'],ascending=False)['DC_POWER']
df_pgen2.groupby('SOURCE_KEY').max().sort_values(by=['AC_POWER'],ascending=False)['AC_POWER']
print('The number of rows with missing values are :',sum(df_pgen1.isnull().values.any(axis=1)))
print('Rows with empty values are :')
print(df_pgen1[df_pgen1.isnull().any(axis=1)])
print('The number of rows with missing values are :',sum(df_sen1.isnull().values.any(axis=1)))
print('Rows with empty values are :')
print(df_sen1[df_sen1.isnull().any(axis=1)])
print('The number of rows with missing values are :',sum(df_pgen2.isnull().values.any(axis=1)))
print('Rows with empty values are :')
print(df_pgen2[df_pgen2.isnull().any(axis=1)])
print('The number of rows with missing values are :',sum(df_sen2.isnull().values.any(axis=1)))
print('Rows with empty values are :')
print(df_sen2[df_sen2.isnull().any(axis=1)])