# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import fbprophet



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Single cell for data loading and pre-processing. Consolidating all relevant steps from the notebook below.



# Step 1 - loading data sets

df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')



# Step 2 - correcting date_time format

df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')



# Step 3 - splitting date and time

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())



df_psense1['DATE'] = df_psense1['DATE_TIME'].apply(lambda x:x.date())

df_psense1['TIME'] = df_psense1['DATE_TIME'].apply(lambda x:x.time())



# Step 4 - correcting data_time format for the DATE column

df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')

df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')



# Step 5 - splitting hour and minutes

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute



df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour

df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.head(50)
df_psense1.head(50)
result_left = pd.merge(df_pgen1,df_psense1, on='DATE_TIME',how='left') 

temp = result_left.groupby(["SOURCE_KEY_x","DATE_x","HOUR_x"]).agg(DC_POWER=("DC_POWER",'mean'))

temptwo = result_left.groupby(["SOURCE_KEY_x", "DATE_x","HOUR_x"]).agg(AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE","mean"))

final = pd.merge(temp,temptwo["AMBIENT_TEMPERATURE"],on="SOURCE_KEY_x",how= 'outer')

final.head(50)
temp.head(50)
temptwo.head(50)
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(result_left.MODULE_TEMPERATURE,

        result_left.DC_POWER,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='DC POWER')



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power vs. Module Temperature')

plt.xlabel('Module Temperature')

plt.ylabel('DC Power')

plt.show()