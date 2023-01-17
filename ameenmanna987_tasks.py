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
df1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df3=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df4=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df1.head()
df2.head()
df3.head()
df4.head()
df1.columns
df2.columns
df3.columns
df4.columns
df1.nunique()
df1['SOURCE_KEY'].unique()
df2.nunique()
df2['SOURCE_KEY'].unique()
df3.nunique()
df3['SOURCE_KEY'].unique()
df4.nunique()
df4['SOURCE_KEY'].unique()
df1.describe()
df2.describe()
df3.describe()
df4.describe()
df1['DAILY_YIELD'].mean()
df3['DAILY_YIELD'].mean()
df2['IRRADIATION'].sum()
df4['IRRADIATION'].sum()
df2.head(5)
df2.info()
df2['DATE_TIME']=pd.to_datetime(df2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
df2.info()
df2['DATE']=pd.to_datetime(df2['DATE_TIME'].dt.date)
df2
d=df2.groupby(['DATE']).sum()
d
#df2.groupby(['DATE']).sum().IRRADIATION
d['IRRADIATION']
df4.head()
df4['DATE_TIME']=pd.to_datetime(df4['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
df4.info()
df4['DATE']=pd.to_datetime(df4['DATE_TIME'].dt.date)
df4
df4.groupby(['DATE']).sum().IRRADIATION
df2['AMBIENT_TEMPERATURE'].max()
df4['AMBIENT_TEMPERATURE'].max()
df2['MODULE_TEMPERATURE'].max()
df4['MODULE_TEMPERATURE'].max()
df1['SOURCE_KEY'].nunique()
df3['SOURCE_KEY'].nunique()
df1
df1.info()
df1['DATE_TIME']=pd.to_datetime(df1['DATE_TIME'],format='%d-%m-%Y %H:%M')
df1.info()
df1['DATE']=pd.to_datetime(df1['DATE_TIME']).dt.date
df1
df1.groupby(['DATE']).max()
df1['DC_POWER'].max()
df1['DC_POWER'].min()
df1['AC_POWER'].max()

df1['AC_POWER'].min()
df3['DC_POWER'].max()
df3['DC_POWER'].min()
df3['AC_POWER'].max()
df3['AC_POWER'].min()
df1[df1['DC_POWER']==df1['DC_POWER'].max()]
#df1[df1['AC_POWER']==df1['AC_POWER'].max()]['SOURCE_KEY']
(df1.iloc[df1['AC_POWER'].idxmax()])['SOURCE_KEY']
(df3.iloc[df3['AC_POWER'].idxmax()])['SOURCE_KEY']
(df3.iloc[df3['DC_POWER'].idxmax()])['SOURCE_KEY']
amn = df1.groupby('SOURCE_KEY').sum()
amn['AC_POWER'].sort_values()
amn # for a particular source key,all of the data is added
