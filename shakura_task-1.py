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

df_wgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_pgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1.describe()
df_wgen1.describe()
df_pgen2.describe()
df_wgen2.describe()
df_pgen1.head()
df_pgen2.tail()
df_wgen1.head()
df_wgen2.head()
df_pgen1['DAILY_YIELD'].mean()
df_pgen2['DAILY_YIELD'].mean()
import pandas as pd
df_wgen1['DATE_TIME']=pd.to_datetime(df_wgen1['DATE_TIME'],format='%Y-%m-%d %H:%M')

df_wgen1['DATE']=df_wgen1['DATE_TIME'].apply(lambda x:x.date())

df_wgen1['TIME']=df_wgen1['DATE_TIME'].apply(lambda x:x.time())

df_wgen1.info()
df_pgen1['DATE_TIME']=pd.to_datetime(df_pgen1['DATE_TIME'],format='%d-%m-%Y %H:%M')

df_pgen1['DATE']=df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME']=df_pgen1['DATE_TIME'].apply(lambda x:x.time())

df_pgen1.info()
df_wgen2['DATE_TIME'] = pd.to_datetime(df_wgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')   

df_wgen2['DATE']=df_wgen2['DATE_TIME'].apply(lambda x:x.date())

df_wgen2['TIME']=df_wgen2['DATE_TIME'].apply(lambda x:x.time())

df_wgen2.info()
print("\n2. The total irradiation per day is:\n",df_wgen1.groupby(['DATE'])['IRRADIATION'].sum())
print("\n2. The total irradiation per day is:\n",df_wgen2.groupby(['DATE'])['IRRADIATION'].sum())
print("\n2. The maximum ambident module temperture is:\n", df_wgen2[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max())
len(df_pgen1['SOURCE_KEY'].unique())
len(df_pgen2['SOURCE_KEY'].unique())
df_pgen1[['DC_POWER','AC_POWER']].max()
df_pgen1[['DC_POWER','AC_POWER']].min()
df_pgen2[['DC_POWER','AC_POWER']].max()
df_pgen2[['DC_POWER','AC_POWER']].min()
print("The maximum dc power is",df_pgen1['DC_POWER'].max(),"produced by the inverter",df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmax()])
print("The maximum ac power is",df_pgen1['AC_POWER'].max(),"produced by the inverter",df_pgen1['SOURCE_KEY'].values[df_pgen1['AC_POWER'].argmax()])
#For every date/day - 34 inverters , 24 hours ,4 times per day

34*24*4
df_pgen1['HOUR']=pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTE']=pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1[df_pgen1['DATE']=='2020-05-21'][df_pgen1['TIME']==7]
df_pgen1[df_pgen1['DATE']=='2020-05-16']['SOURCE_KEY'].value_counts()
ideal_values=34*24*4

values = df_pgen1['SOURCE_KEY'].value_counts()

missing = (ideal_values)*22 - values.sum()

print(missing)