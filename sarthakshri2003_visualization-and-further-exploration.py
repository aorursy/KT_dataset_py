# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv');

df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv');

df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv');

df_psense2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv');

#loading the csv files into the dataframes
# Correcting date_time format

df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')



# Splitting date and time

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())



df_psense1['DATE'] = df_psense1['DATE_TIME'].apply(lambda x:x.date())

df_psense1['TIME'] = df_psense1['DATE_TIME'].apply(lambda x:x.time())





# Correcting data_time format for the DATE column

df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')

df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')



# Splitting hour and minutes

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute



df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour

df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.head()
df_psense1.head()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_pgen1['DC_POWER'],

        df_pgen1['AC_POWER'],

        marker='o',

        linestyle='',

        alpha=.5, #transparency

        ms=3, #size of the dot

        label='Correlation Between DC Power & AC Power')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Correlation Between DC Power & AC Power')

plt.xlabel('AC_POWER')

plt.ylabel('DC_POWER')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(df_psense1['HOUR'],

        df_psense1['IRRADIATION'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='Irradiation With Time')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Time')

plt.xlabel('Hour')

plt.ylabel('Irradiation')

plt.show()
dates = df_psense1["DATE"].unique()



_, ax = plt.subplots(1, 1, figsize=(18,9))



for date in dates:

    df_data = df_psense1[df_psense1["DATE"] == date]

    

    ax.plot(df_data.HOUR,

            df_data.AMBIENT_TEMPERATURE,

            marker='o',

            linestyle='',

            alpha=.5,

            ms=10,

            label=pd.to_datetime(date,format='%Y-%m-%d').date()

            )

    

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Ambient Tempreture vs time in hours for each day')

plt.xlabel('HOURS')

plt.ylabel('Ambient Tempreture')

plt.show()



#the ambient temperature is plotted for each day in a different color, and is plotted against the hour in the day
dates = df_psense1["DATE"].unique()



_, ax = plt.subplots(1, 1, figsize=(18,9))



for date in dates:

    df_data = df_psense1[df_psense1["DATE"] == date]

    

    ax.plot(df_data.HOUR,

            df_data.MODULE_TEMPERATURE,

            marker='o',

            linestyle='',

            alpha=.5,

            ms=10,

            label=pd.to_datetime(date,format='%Y-%m-%d').date()

            )

    

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Module Tempreture vs time in hours for each day')

plt.xlabel('HOURS')

plt.ylabel('Module Tempreture')

plt.show()



#the module temperature is plotted for each day in a different color, and is plotted against the hour in the day
_, ax = plt.subplots(1, 1, figsize=(24, 10))



ax.plot(df_pgen1['DATE'],

        df_pgen1['TOTAL_YIELD'],

        marker='o',

        linestyle='',

        alpha=.75,

        ms=5,

        label='Yield With Date')

ax.grid()

ax.margins(0.025)

ax.legend()

plt.title('Total Yield per Day')

plt.xlabel('DATE')

plt.ylabel('TOTAL_YIELD')

plt.show()
dates = df_psense1['DATE'].unique()



_, ax = plt.subplots(1, 1, figsize=(18, 9))



for date in dates:

    df_data = df_psense1[df_psense1['DATE']==date]#[df_psense1['IRRADIATION']>0]



    ax.plot(df_data.AMBIENT_TEMPERATURE,

            df_data.MODULE_TEMPERATURE,

            marker='o',

            linestyle='',

            alpha=.5,

            ms=10,

            label=pd.to_datetime(date,format='%Y-%m-%d').date()

           )



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Module Temperature vs. Ambient Temperature')

plt.xlabel('Ambient Temperature')

plt.ylabel('Module Temperature')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_psense1['IRRADIATION'],

        df_psense1['MODULE_TEMPERATURE'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='module temperature')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Module Tempreture')

plt.xlabel('Irradiation')

plt.ylabel('Module Tempreture')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(df_psense1['IRRADIATION'],

        df_psense1['AMBIENT_TEMPERATURE'],

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='ambient temperature')

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Irradiation vs. Ambient Tempreture')

plt.xlabel('Irradiation')

plt.ylabel('Ambient Tempreture')

plt.show()
result_left = pd.merge(df_pgen1,df_psense1, on='DATE_TIME',how='left')
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(result_left.IRRADIATION,

        result_left.DC_POWER,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='DC POWER')



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('DC Power vs. Irradiation')

plt.xlabel('Irradiation')

plt.ylabel('DC Power')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(result_left.IRRADIATION,

        result_left.AC_POWER,

        marker='o',

        linestyle='',

        alpha=.5,

        ms=10,

        label='AC POWER')



ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('AC Power vs. Irradiation')

plt.xlabel('Irradiation')

plt.ylabel('AC Power')

plt.show()