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
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df1.describe()
df1.min()
df2.describe()
df2.count()
df2.max()
df1.head()
df2.tail(43)
df2.head(60)
dymean= df1['DAILY_YIELD'].mean()

print(dymean)
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M')   

df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date   

df1['DATE'] = pd.to_datetime(df1['DATE'])  

df1.info()
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')  

df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date

df2['DATE'] = pd.to_datetime(df2['DATE'])

df2.info()
print('Total Irridation per day:')

df1i = df2.groupby([df2['DATE']])['IRRADIATION'].sum()

df1i
am=df2['AMBIENT_TEMPERATURE'].max()

am
mm=df2['MODULE_TEMPERATURE'].max()

mm
in1= df1['SOURCE_KEY'].nunique()

in2= df2['SOURCE_KEY'].nunique()

print(in1)

print(in2)
dmi = df1.groupby('DATE')["DC_POWER"].min()

dmi
dma = df1.groupby('DATE')["DC_POWER"].max()

dma
ami = df1.groupby('DATE')["AC_POWER"].min()

ami
ama = df1.groupby('DATE')["AC_POWER"].max()

ama
df1[df1['DC_POWER'] == df1['DC_POWER'].max()]['SOURCE_KEY']
rankdc=df1.groupby(['SOURCE_KEY']).count()

rank[['DC_POWER']].rank()
rankac=df1.groupby(['SOURCE_KEY']).count()

rank[['AC_POWER']].rank()
#Total data which should be there for every day

22*4*24
#Data which is actually there for every day

df1['DATE'].value_counts()