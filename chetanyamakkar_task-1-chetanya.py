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

df1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df2=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df1.columns
df2.columns
df1.shape      # (rows,collums)
df2.shape      # (rows,collums)   
df1.describe()      # Exploring Datasets
df2.describe()      # Exploring Datasets
df1['DAILY_YIELD'].mean()      # Plant 1
import pandas as pd

import datetime as dt

df2['DATE_TIME']=pd.to_datetime(df2['DATE_TIME'])      # Changing to datetime
df2['Date']=df2['DATE_TIME'].dt.date

ir=df2.groupby(['Date']).sum()

ir['IRRADIATION']      # Total Irradiation Per Day

df2['AMBIENT_TEMPERATURE'].max()
df2['MODULE_TEMPERATURE'].max()
df1['SOURCE_KEY'].nunique()      # No. of Inverters Plant 1
df2['SOURCE_KEY'].nunique()
import pandas as pd

import datetime as dt

df1['DATE_TIME']=pd.to_datetime(df1['DATE_TIME'])
df1['Date']=df1['DATE_TIME'].dt.date
max_dc=df1.groupby(['Date']).max()      # Per day dc max

max_dc['DC_POWER']
min_dc=df1.groupby(['Date']).min()      # Per day min ac

min_dc['DC_POWER']
max_ac=df1.groupby(['Date']).max()      # Per day max ac

max_ac['AC_POWER']
min_ac=df1.groupby(['Date']).min()      # per day min ac

min_ac['AC_POWER']
df1[df1['DC_POWER']==df1['DC_POWER'].max()]['SOURCE_KEY']
df1[df1['AC_POWER']==df1['AC_POWER'].max()]['SOURCE_KEY']
rank=df1.groupby(['SOURCE_KEY']).count()

rank[['DC_POWER','AC_POWER']].rank()
#yes

22*4*24
df1
df1['Date'].value_counts()