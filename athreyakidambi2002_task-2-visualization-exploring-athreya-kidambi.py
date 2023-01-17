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
df_Plant1spgd = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_Plant1wsd = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_Plant2spgd = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_Plant2wsd = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
#Convert the DATE_TIME column to the datetime format
df_Plant1spgd['DATE_TIME'] = pd.to_datetime(df_Plant1spgd['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_Plant1wsd['DATE_TIME'] = pd.to_datetime(df_Plant1wsd['DATE_TIME'],format = '%Y-%m-%d %H:%M')

#Convert the DATE_TIME column to the datetime format
df_Plant2spgd['DATE_TIME'] = pd.to_datetime(df_Plant2spgd['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_Plant2wsd['DATE_TIME'] = pd.to_datetime(df_Plant2wsd['DATE_TIME'],format = '%Y-%m-%d %H:%M')
#Split the Date and Time for Plant 1
df_Plant1spgd['DATE'] = df_Plant1spgd['DATE_TIME'].apply(lambda x:x.date())
df_Plant1spgd['TIME'] = df_Plant1spgd['DATE_TIME'].apply(lambda x:x.time())

df_Plant1wsd['DATE'] = df_Plant1wsd['DATE_TIME'].apply(lambda x:x.date())
df_Plant1wsd['TIME'] = df_Plant1wsd['DATE_TIME'].apply(lambda x:x.time())

#Split the Date and Time for Plant 2
df_Plant2spgd['DATE'] = df_Plant2spgd['DATE_TIME'].apply(lambda x:x.date())
df_Plant2spgd['TIME'] = df_Plant2spgd['DATE_TIME'].apply(lambda x:x.time())

df_Plant2wsd['DATE'] = df_Plant2wsd['DATE_TIME'].apply(lambda x:x.date())
df_Plant2wsd['TIME'] = df_Plant2wsd['DATE_TIME'].apply(lambda x:x.time())
#Correct the format of the DATE column for Plant 1
df_Plant1spgd['DATE'] = pd.to_datetime(df_Plant1spgd['DATE'],format = '%Y-%m-%d')
df_Plant1wsd['DATE'] = pd.to_datetime(df_Plant1wsd['DATE'],format = '%Y-%m-%d')

#Correct the format of the DATE column for Plant 2
df_Plant2spgd['DATE'] = pd.to_datetime(df_Plant2spgd['DATE'],format = '%Y-%m-%d')
df_Plant2wsd['DATE'] = pd.to_datetime(df_Plant2wsd['DATE'],format = '%Y-%m-%d')
#Split the Hour and Minutes for easier analysis for Plant 1
df_Plant1spgd['HOUR'] = pd.to_datetime(df_Plant1spgd['TIME'],format='%H:%M:%S').dt.hour
df_Plant1spgd['MINUTES'] = pd.to_datetime(df_Plant1spgd['TIME'],format='%H:%M:%S').dt.minute

df_Plant1wsd['HOUR'] = pd.to_datetime(df_Plant1wsd['TIME'],format='%H:%M:%S').dt.hour
df_Plant1wsd['MINUTES'] = pd.to_datetime(df_Plant1wsd['TIME'],format='%H:%M:%S').dt.minute

#Split the Hour and Minutes for easier analysis for Plant 2
df_Plant2spgd['HOUR'] = pd.to_datetime(df_Plant2spgd['TIME'],format='%H:%M:%S').dt.hour
df_Plant2spgd['MINUTES'] = pd.to_datetime(df_Plant2spgd['TIME'],format='%H:%M:%S').dt.minute

df_Plant2wsd['HOUR'] = pd.to_datetime(df_Plant2wsd['TIME'],format='%H:%M:%S').dt.hour
df_Plant2wsd['MINUTES'] = pd.to_datetime(df_Plant2wsd['TIME'],format='%H:%M:%S').dt.minute
#Imprting Matplotlib Pyplot and Seaborn libraries for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#Plotting the Scatter Chart for DC Power vs Date and time and categorising by Date for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.DC_POWER,#.rolling(window=20).mean(),
        label='DC POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('DC Power')
plt.show()
#Plotting Scatter Chart for DC Power vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.DC_POWER,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_Power vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('DC_Power')
plt.show()
#Plotting the Scatter Chart for DC Power vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.DC_POWER,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Date and Time for 34 days Scatter Plot sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('DC_Power')
plt.show()
#Plotting Line Graph for the AC Power vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.AC_POWER,#.rolling(window=20).mean(),
        label='AC POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('AC Power')
#Plotting Scatter Chart for DC Power vs Date and Time for 34 days for Plant 1plt.show()
#Plotting Scatter Chart for AC Power vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.AC_POWER,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC_POWER')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_Power vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('AC_Power')
plt.show()
#Plotting the Scatter Chart for AC Power vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.AC_POWER,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('AC_Power')
plt.show()
#Plotting Line Graph for the Daily Yield vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.DAILY_YIELD,#.rolling(window=20).mean(),
        label='DAILY YIELD'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting Scatter Chart for Daily Yield vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.DAILY_YIELD,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY_YIELD')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting the Scatter Chart for Daily Yield vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.DAILY_YIELD,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting Line Graph for the Total Yield vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.TOTAL_YIELD,#.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Total  Yield')
plt.show()
#Plotting Scatter Chart for Total Yield vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DATE_TIME,
        df_Plant1spgd.TOTAL_YIELD,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Total Yield')
plt.show()
#Plotting the Scatter Chart for Total Yield vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.TOTAL_YIELD,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Total Yield')
plt.show()
#Plotting Line Graph for the Ambient Temperature vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
        label='AMBIENT TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature vs Date and Time for 34 Days Line Graph for Plant 1 ')
plt.xlabel('Date and Time')
plt.ylabel('Ambient Temperature')
plt.show()
#Plotting Scatter Chart for Ambient Temperature vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AMBIENT TEMPERATURE (centigrade)')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature vs Date and Time for 34 days Scatter Chart for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Ambient Tempreture')
plt.show()
#Plotting the Scatter Chart for Ambient Temperature vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]#[df_Plant1wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Ambient Temperature')
plt.show()
#Plotting Line Graph for the Module Temperature vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
        label='MODULE TEMPERATURE (centigrade)'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting Scatter Chart for Module Temperature vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE (centigrade)')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting the Scatter Chart for Module Temperature vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]#[df_Plant1wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting Line Graph for the Irradiation vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.IRRADIATION,#.rolling(window=20).mean(),
        label='IRRADIATION'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
#Plotting Scatter Chart for Irradiation vs Date and Time for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.DATE_TIME,
        df_Plant1wsd.IRRADIATION,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='IRRADIATION')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
#Plotting the Scatter Chart for DC Power vs Date and time and categorising by Date for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]#[df_Plant1wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.IRRADIATION,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 days Scatter Plot for Plant 1 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
## Plotting the Line Graph for DC Power vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.DC_POWER,#.rolling(window=20).mean(),
        label='DC POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Date and Time for 34 days Line Graph for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('DC Power')
plt.show()
#Plotting the Scatter Chart for DC Power vs Date and time and categorising by Date for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.DC_POWER,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_Power vs Date and Time for 34 days Scatter Plot for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('DC_Power')
plt.show()
#Plotting the Scatter Chart for DC Power vs Date and time and categorising by Date for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.DC_POWER,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('DC_Power')
plt.show()
#Plotting the Line Graph for AC Power vs date and time for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.AC_POWER,#.rolling(window=20).mean(),
        label='AC POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Date and Time for 34 days Line Graph for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('AC Power')
plt.show()
#Plotting the Scatter Chart for AC Power vs date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.AC_POWER,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_Power vs Date and Time for 34 days Scatter Plot for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('AC_Power')
plt.show()
#Plotting the Scatter Chart for AC Power vs Date and time and categorising by Date for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.AC_POWER,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('AC_Power')
plt.show()
#Plotting the Scatter Chart for Daily Yield vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.DAILY_YIELD,#.rolling(window=20).mean(),
        label='DAILY YIELD'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting the Scatter Chart for Daily Yield vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.DAILY_YIELD,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Scatter Plot for Plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting the Scatter Chart for Daily Yield vs Date and time and categorising by Date for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.DAILY_YIELD,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Daily Yield vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Daily Yield')
plt.show()
#Plotting the Line Graph for Total Yield vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.TOTAL_YIELD,#.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Line Graph for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Total  Yield')
plt.show()
#Plotting the Scatter Chart for Daily Yield vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DATE_TIME,
        df_Plant2spgd.TOTAL_YIELD,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Scatter Plot for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Total Yield')
plt.show()
#Plotting the Scatter Chart for Total Yield vs Date and time and categorising by Date for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DATE_TIME,
            df_data.TOTAL_YIELD,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Total Yield vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Total Yield')
plt.show()
#Plotting the Line Graph for Ambient Temperature vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
        label='AMBIENT TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature vs Date and Time for 34 Days Line Graph for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Ambient Temperature')
plt.show()
#Plotting the Scatter Plot for Ambient Temperature vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AMBIENT TEMPERATURE')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs Date and Time for 34 days Scatter Plot  for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Module Tempreture')
plt.show()
#Plotting the Scatter Plot for Ambient Temperature vs Date and time for 34 days and categorising by date for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]#[df_Plant2wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.AMBIENT_TEMPERATURE,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Ambient Temperature')
plt.show()
#Plotting the Line Graph for Module Temperature vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
        label='MODULE TEMPERATURE'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 Days Line Graph for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting the Scatter Plot for Module Temperature vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 days Scatter Plot for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting the Scatter Plot for Module Temperature vs Date and time for 34 days and categorising by date for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]#[df_Plant2wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.MODULE_TEMPERATURE,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Module Temperature')
plt.show()
#Plotting the Line Graph for Irradiation vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.IRRADIATION,#.rolling(window=20).mean(),
        label='IRRADIATION'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 Days Line Graph for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
#Plotting the Scatter Plot for Irradiation vs Date and time for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.DATE_TIME,
        df_Plant2wsd.IRRADIATION,#.rolling(window=20).mean(),
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='IRRADIATION')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 days Scatter Plot for Plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
#Plotting the Scatter Plot for Irradiation vs Date and time for 34 days and categorising by date for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]#[df_Plant2wsd['IRRADIATION']>0]

    ax.plot(df_data.DATE_TIME,
            df_data.IRRADIATION,#.rolling(window=20).mean(),
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs Date and Time for 34 days Scatter Plot for Plant 2 sorted by date')
plt.xlabel('Date and Time')
plt.ylabel('Irradiation')
plt.show()
#Line Graph to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1wsd.AMBIENT_TEMPERATURE,
        df_Plant1wsd.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='MODULE TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.AMBIENT_TEMPERATURE,
        df_Plant1wsd.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]#[df_Plant1wsd['IRRADIATION']>0]

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
plt.title('Scatter Plot sorted by date showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Line Graph to visualize how AC Power varies with DC Power for Plant 1 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1spgd.DC_POWER/10,
        df_Plant1spgd.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with DC Power for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DC_POWER/10,
        df_Plant1spgd.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with DC Power for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DC_POWER/10,
            df_data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how Module Temperature varies with Irradiation for Plant 1 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1wsd.IRRADIATION,
        df_Plant1wsd.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='MODULE TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Module Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart to visualize how Module Temperature varies with Irradiation for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.IRRADIATION,
        df_Plant1wsd.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Module Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Module Temperature varies with Irradiation for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]

    ax.plot(df_data.IRRADIATION,
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
plt.title('Scatter Plot sorted by date showing How Module Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Line Graph to visualize how Ambient Temperature varies with Irradiation for Plant 1 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1wsd.IRRADIATION,
        df_Plant1wsd.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='AMBIENT TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Scatter Chart to visualize how Ambient Temperature varies with Irradiation for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1wsd.IRRADIATION,
        df_Plant1wsd.AMBIENT_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AMBIENT TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Ambient Temperature varies with Irradiation for 34 days for Plant 1

dates = df_Plant1wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1wsd[df_Plant1wsd['DATE']==date]

    ax.plot(df_data.IRRADIATION,
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
plt.title('Scatter Plot sorted by date showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Line Graph to visualize how Total Yield varies with Daily Yield for Plant 1 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant1spgd.DAILY_YIELD,
        df_Plant1spgd.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Daily Yield for 34 Days for Plant 1')
plt.xlabel('Daily Yield')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Daily Yield for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant1spgd.DAILY_YIELD,
        df_Plant1spgd.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Daily Yield for 34 Days for Plant 1')
plt.xlabel('Daily Yield')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Daily_Yield for 34 days for Plant 1

dates = df_Plant1spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant1spgd[df_Plant1spgd['DATE']==date]

    ax.plot(df_data.DAILY_YIELD,
            df_data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Daily_Yield for 34 Days for Plant 1')
plt.xlabel('Daily_Yield')
plt.ylabel('Total Yield')
plt.show()
result_outer1 = pd.merge(df_Plant1spgd,df_Plant1wsd,on='DATE_TIME',how='outer')
#To merge the Data Frames of Solar Power Generation Data and Weather Sensor Data for Plant 1
result_outer2 = pd.merge(df_Plant2spgd,df_Plant2wsd, on='DATE_TIME',how='outer')
#To merge the Data Frames of Solar Power Generation Data and Weather Sensor Data for Plant 2
#Line Graph to visualize how DC Power varies with Irradiation for Plant 1 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.IRRADIATION,
        result_outer1.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()

#Scatter Chart to visualize how DC Power varies with Irradiation for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()


#Scatter Chart categorised by date to visualize how DC Power varies with Irradiation for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Irradiation for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.IRRADIATION,
        result_outer1.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()


#Scatter Chart to visualize how AC Power varies with Irradiation for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Temperature')
plt.ylabel('AC Power')
plt.show()

#Scatter Chart categorised by date to visualize how AC Power varies with Irradiation for 34 days for Plant 2


dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()

#Line Graph to visualize how Daily Yield varies with Irradiation for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.IRRADIATION,
        result_outer1.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Irradiation for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Irradiation for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Irradiation for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.IRRADIATION,
        result_outer1.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Irradiation for 34 days for Plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Irradiation for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Irradiation for 34 Days for Plant 1')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how DC Power varies with Module Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart to visualize how DC Power varies with Module Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart categorised by date to visualize how DC Power varies with Module Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Module Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with Module Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with Module Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how DC Power varies with Ambient Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart to visualize how DC Power varies with Ambient Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart categorised by date to visualize how DC Power varies with Ambient Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Ambient Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with Ambient Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with Ambient Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how Daily Yield varies with Module Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Module Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Module Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Module Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Module Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Module Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Module Temperature for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with Ambient Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Ambient Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Ambient Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Ambient Temperature for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Ambient Temperature for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AMBIENT_TEMPERATURE,
        result_outer1.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Ambient Temperature for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 1')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with DC Power for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.DC_POWER,
        result_outer1.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with DC Power for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.DC_POWER,
        result_outer1.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with DC Power for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.DC_POWER,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with DC Power for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.DC_POWER,
        result_outer1.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with DC Power for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.DC_POWER,
        result_outer1.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with DC Power for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.DC_POWER,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with DC Power for 34 Days for Plant 1')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with AC Power for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AC_POWER,
        result_outer1.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('AC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with AC Power for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AC_POWER,
        result_outer1.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with AC Power for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AC_POWER,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('AC Power')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with AC Power for Plant 1 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer1.AC_POWER,
        result_outer1.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('AC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with AC Power for 34 days for Plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.AC_POWER,
        result_outer1.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with AC Power for 34 days for Plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)&(result_outer1['IRRADIATION']>0)]

    ax.plot(data.AC_POWER,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with AC Power for 34 Days for Plant 1')
plt.xlabel('AC Power')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2wsd.AMBIENT_TEMPERATURE,
        df_Plant2wsd.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='MODULE TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.AMBIENT_TEMPERATURE,
        df_Plant2wsd.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Module Temperature varies with Ambient Temperature for 34 days for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]#[df_Plant2wsd['IRRADIATION']>0]

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
plt.title('Scatter Plot sorted by date showing How Module Temperature varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#Line Graph to visualize how AC Power varies with DC Power for Plant 2 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2spgd.DC_POWER/10,
        df_Plant2spgd.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with DC Power for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DC_POWER,
        df_Plant2spgd.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with DC Power for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DC_POWER,
            df_data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how Module Temperature varies with Irradiation for Plant 2 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2wsd.IRRADIATION,
        df_Plant2wsd.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='MODULE TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Module Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart to visualize how Module Temperature varies with Irradiation for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.IRRADIATION,
        df_Plant2wsd.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='MODULE TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Module Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Module Temperature varies with Irradiation for 34 days for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]

    ax.plot(df_data.IRRADIATION,
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
plt.title('Scatter Plot sorted by date showing How Module Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#Line Graph to visualize how Ambient Temperature varies with Irradiation for Plant 2 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2wsd.IRRADIATION,
        df_Plant2wsd.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='AMBIENT TEMPERATURE'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Scatter Chart to visualize how Ambient Temperature varies with Irradiation for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2wsd.IRRADIATION,
        df_Plant2wsd.AMBIENT_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AMBIENT TEMPERATURE'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Scatter Chart categorised by date to visualize how Ambient Temperature varies with Irradiation for 34 days for Plant 2

dates = df_Plant2wsd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2wsd[df_Plant2wsd['DATE']==date]

    ax.plot(df_data.IRRADIATION,
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
plt.title('Scatter Plot sorted by date showing How Ambient Temperature varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperature')
plt.show()
#Line Graph to visualize how Total Yield varies with Daily Yield for Plant 2 for 34 days

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_Plant2spgd.DAILY_YIELD,
        df_Plant2spgd.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Daily Yield for 34 Days for Plant 2')
plt.xlabel('Daily Yield')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Daily Yield for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_Plant2spgd.DAILY_YIELD,
        df_Plant2spgd.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD'
        )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Daily Yield for 34 Days for Plant 2')
plt.xlabel('Daily Yield')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Daily_Yield for 34 days for Plant 2

dates = df_Plant2spgd['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_Plant2spgd[df_Plant2spgd['DATE']==date]

    ax.plot(df_data.DAILY_YIELD,
            df_data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Daily_Yield for 34 Days for Plant 2')
plt.xlabel('Daily_Yield')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how DC Power varies with Irradiation for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.IRRADIATION,
        result_outer2.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart to visualize how DC Power varies with Irradiation for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.IRRADIATION,
        result_outer2.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart categorised by date to visualize how DC Power varies with Irradiation for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Irradiation for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.IRRADIATION,
        result_outer2.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with Irradiation for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.IRRADIATION,
        result_outer2.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with Irradiation for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how Daily Yield varies with Irradiation for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.IRRADIATION,
        result_outer2.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Irradiation for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.IRRADIATION,
        result_outer2.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Irradiation for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Irradiation for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.IRRADIATION,
        result_outer2.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Irradiation for 34 days for Plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.IRRADIATION,
        result_outer2.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Irradiation for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.IRRADIATION,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Irradiation for 34 Days for Plant 2')
plt.xlabel('Irradiation')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how DC Power varies with Module Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart to visualize how DC Power varies with Module Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart categorised by date to visualize how DC Power varies with Module Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Module Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with Module Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with Module Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how DC Power varies with Ambient Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.DC_POWER.rolling(window=20).mean(),
        label='DC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How DC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart to visualize how DC Power varies with Ambient Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How DC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Scatter Chart categorised by date to visualize how DC Power varies with Ambient Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How DC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('DC Power')
plt.show()
#Line Graph to visualize how AC Power varies with Ambient Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.AC_POWER.rolling(window=20).mean(),
        label='AC POWER'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How AC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart to visualize how AC Power varies with Ambient Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Scatter Chart categorised by date to visualize how AC Power varies with Ambient Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.AC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How AC Power varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('AC Power')
plt.show()
#Line Graph to visualize how Daily Yield varies with Module Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Module Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Module Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Module Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Module Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Module Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.MODULE_TEMPERATURE,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Module Temperature for 34 Days for Plant 2')
plt.xlabel('Module Temperature')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with Ambient Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with Ambient Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with Ambient Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with Ambient Temperature for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with Ambient Temperature for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AMBIENT_TEMPERATURE,
        result_outer2.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with Ambient Temperature for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AMBIENT_TEMPERATURE,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with Ambient Temperature for 34 Days for Plant 2')
plt.xlabel('Ambient Temperature')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with DC Power for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.DC_POWER,
        result_outer2.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with DC Power for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.DC_POWER,
        result_outer2.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with DC Power for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.DC_POWER,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with DC Power for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.DC_POWER,
        result_outer2.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with DC Power for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.DC_POWER,
        result_outer2.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with DC Power for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.DC_POWER,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with DC Power for 34 Days for Plant 2')
plt.xlabel('DC Power')
plt.ylabel('Total Yield')
plt.show()
#Line Graph to visualize how Daily Yield varies with AC Power for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AC_POWER,
        result_outer2.DAILY_YIELD.rolling(window=20).mean(),
        label='DAILY YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Daily Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart to visualize how Daily Yield varies with AC Power for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AC_POWER,
        result_outer2.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DAILY YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Daily Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Daily Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Daily Yield varies with AC Power for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AC_POWER,
            data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Daily Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Daily Yield')
plt.show()
#Line Graph to visualize how Total Yield varies with AC Power for Plant 2 for 34 days


_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(result_outer2.AC_POWER,
        result_outer2.TOTAL_YIELD.rolling(window=20).mean(),
        label='TOTAL YIELD'
       )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Line Graph showing How Total Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart to visualize how Total Yield varies with AC Power for 34 days for Plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.AC_POWER,
        result_outer2.TOTAL_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='TOTAL YIELD')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How Total Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Total Yield')
plt.show()
#Scatter Chart categorised by date to visualize how Total Yield varies with AC Power for 34 days for Plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)&(result_outer2['IRRADIATION']>0)]

    ax.plot(data.AC_POWER,
            data.TOTAL_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot sorted by date showing How Total Yield varies with AC Power for 34 Days for Plant 2')
plt.xlabel('AC Power')
plt.ylabel('Total Yield')
plt.show()