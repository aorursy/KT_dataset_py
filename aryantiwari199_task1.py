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
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv') #plant1
df1
type(df1)
df1.describe()
df1.min()
df1.max()
df1.count()
df1.shape
len(df1)
df1.head()
df1.tail()
df1.info
df1.sum()
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv') #plant2
df2
type(df2)
df2.shape
df2.describe()
df2.median()
df2.mode()
len(df2)
df2.min()
df2.max()
#QNA

#What is the mean value of daily yield?

df1['DAILY_YIELD'].mean()

df2['DAILY_YIELD'].mean()
df1w = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df1w
#What is the total irradiation per day?

df1w['IRRADIATION'].sum()

#What is the max ambient and module temperature?

df1w['AMBIENT_TEMPERATURE'].max()
df1w['MODULE_TEMPERATURE'].max()
df1
#How many inverters are there for each plant?

df1['SOURCE_KEY'].count()
df2['SOURCE_KEY'].count()
df2w = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df2w
df1
#What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?

da = df1[['DATE_TIME','AC_POWER','DC_POWER']]

asa = da.groupby('DATE_TIME').agg({'AC_POWER':['max','min'],'DC_POWER':['max','min']})

asa
#PLANT2

da2 = df2[['DATE_TIME','AC_POWER','DC_POWER']]

asa2 = da.groupby('DATE_TIME').agg({'AC_POWER':['max','min'],'DC_POWER':['max','min']})

asa2
#Which inverter (source_key) has produced maximum DC/AC power?

dcac1 = df1[['SOURCE_KEY','AC_POWER','DC_POWER']]

acdc1 = dcac1.groupby('SOURCE_KEY').agg({'AC_POWER':['max'],'DC_POWER':['max']})

acdc1.max()
dcac2 = df2[['SOURCE_KEY','AC_POWER','DC_POWER']]

acdc2 = dcac2.groupby('SOURCE_KEY').agg({'AC_POWER':['max'],'DC_POWER':['max']})

acdc2.max()
#Ranking the inverters based on the DC/AC power they produce

dcac1.sort_values( by=['AC_POWER','DC_POWER'], ascending=False)
#Ranking the inverters based on the DC/AC power they produce

dcac2.sort_values( by=['AC_POWER','DC_POWER'], ascending=False)
#FINDING MISSING DATA

df1.isnull()
df2.isnull()
df1.notnull()
df2.notnull()
df1.fillna(0)
df2.fillna(0)