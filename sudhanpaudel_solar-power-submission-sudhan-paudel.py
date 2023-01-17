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
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_psense1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_psense2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_psense1['DATE_TIME'] = pd.to_datetime(df_psense1['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df_pgen1['DATE'] = df_pgen1['DATE_TIME'].dt.date
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].dt.time

df_psense1['DATE'] = df_psense1['DATE_TIME'].dt.date
df_psense1['TIME'] = df_psense1['DATE_TIME'].dt.time



df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df_psense2["DATE_TIME"]=pd.to_datetime(df_psense2["DATE_TIME"], format='%Y-%m-%d %H:%M:%S') 

df_pgen2['DATE'] = df_pgen2['DATE_TIME'].dt.date
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].dt.time

df_psense2['DATE'] = df_psense2['DATE_TIME'].dt.date
df_psense2['TIME'] = df_psense2['DATE_TIME'].dt.time
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_psense1['DATE'] = pd.to_datetime(df_psense1['DATE'],format = '%Y-%m-%d')
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_psense2['DATE'] = pd.to_datetime(df_psense2['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_psense1['HOUR'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.hour
df_psense1['MINUTES'] = pd.to_datetime(df_psense1['TIME'],format='%H:%M:%S').dt.minute

df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute

df_psense2['HOUR'] = pd.to_datetime(df_psense2['TIME'],format='%H:%M:%S').dt.hour
df_psense2['MINUTES'] = pd.to_datetime(df_psense2['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.head()
df_pgen2.head(50)
df_psense1.head()
df_psense2.head()
import matplotlib.pyplot as plt
import seaborn as sns
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()
print ("minimum="+ str(df_psense1.AMBIENT_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense1.AMBIENT_TEMPERATURE.max()) )
print ("mean="+ str(df_psense1.AMBIENT_TEMPERATURE.mean()) )
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2.DATE_TIME,
        df_psense2.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()
print ("minimum="+ str(df_psense2.AMBIENT_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense2.AMBIENT_TEMPERATURE.max()) )
print ("mean="+ str(df_psense2.AMBIENT_TEMPERATURE.mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

print ("minimum="+ str(df_psense1.MODULE_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense1.MODULE_TEMPERATURE.max()) )
print ("mean="+ str(df_psense1.MODULE_TEMPERATURE.mean()) )
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2.DATE_TIME,
        df_psense2.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

print ("minimum="+ str(df_psense2.MODULE_TEMPERATURE.min()) )
print ("maximum="+ str(df_psense2.MODULE_TEMPERATURE.max()) )
print ("mean="+ str(df_psense2.MODULE_TEMPERATURE.mean()) )
#FOR plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.plot(df_psense1.DATE_TIME,
        df_psense1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module'
       )

ax.plot(df_psense1.DATE_TIME,
        (df_psense1.MODULE_TEMPERATURE-df_psense1.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature and Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

#For plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2.DATE_TIME,
        df_psense2.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient'
       )

ax.plot(df_psense2.DATE_TIME,
        df_psense2.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module'
       )

ax.plot(df_psense2.DATE_TIME,
        (df_psense2.MODULE_TEMPERATURE-df_psense1.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Temperature and Module Temperature over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.show()

# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str(df_pgen1.AC_POWER.min()) )
print ("maximum="+ str(df_pgen1.AC_POWER.max()) )
print ("mean="+ str(df_pgen1.AC_POWER.mean()) )
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen2.DATE_TIME,
        df_pgen2.AC_POWER.rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str(df_pgen2.AC_POWER.min()) )
print ("maximum="+ str(df_pgen2.AC_POWER.max()) )
print ("mean="+ str(df_pgen2.AC_POWER.mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str((df_pgen1.DC_POWER/10).min()) )
print ("maximum="+ str((df_pgen1.DC_POWER/10).max()) )
print ("mean="+ str((df_pgen1.DC_POWER/10).mean()) )
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen2.DATE_TIME,
        (df_pgen2.DC_POWER).rolling(window=500).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC_POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()

print ("minimum="+ str((df_pgen2.DC_POWER).min()) )
print ("maximum="+ str((df_pgen2.DC_POWER).max()) )
print ("mean="+ str((df_pgen2.DC_POWER).mean()) )
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DATE_TIME,
        df_pgen1.AC_POWER.rolling(window=500).mean(),
        label='AC'
       )

ax.plot(df_pgen1.DATE_TIME,
       (df_pgen1.DC_POWER/10).rolling(window=500).mean(),
        label='DC'
       )

ax.plot(df_pgen1.DATE_TIME,
       ((df_pgen1.DC_POWER/10)-df_pgen1.AC_POWER).rolling(window=500).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DC POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('kW')
plt.show()

#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen2.DATE_TIME,
        df_pgen2.AC_POWER.rolling(window=500).mean(),
        label='AC'
       )

ax.plot(df_pgen2.DATE_TIME,
        df_pgen2.DC_POWER.rolling(window=500).mean(),
        label='DC'
       )

ax.plot(df_pgen2.DATE_TIME,
       (df_pgen2.DC_POWER-df_pgen2.AC_POWER).rolling(window=500).mean(),
        label='Difference'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DC POWER over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('kW')
plt.show()

#for plant 1

df_data = df_psense1[df_psense1['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.MODULE_TEMPERATURE,
        label="MODULE"
       )
ax.plot(df_data.DATE_TIME,
        df_data.AMBIENT_TEMPERATURE,
        label="AMBIENT"
       )

ax.plot(df_data.DATE_TIME,
      ((df_data.MODULE_TEMPERATURE)-(df_data.AMBIENT_TEMPERATURE)),
        label="Difference"
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Module Temperature with Ambient temperature on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
#for plant 2

df_data = df_psense2[df_psense2['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.MODULE_TEMPERATURE,
        label="MODULE"
       )
ax.plot(df_data.DATE_TIME,
        df_data.AMBIENT_TEMPERATURE,
        label="AMBIENT"
       )

ax.plot(df_data.DATE_TIME,
      ((df_data.MODULE_TEMPERATURE)-(df_data.AMBIENT_TEMPERATURE)),
        label="Difference"
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Module Temperature with Ambient temperature on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1.DATE_TIME,
        df_psense1.IRRADIATION.rolling(window=40).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('IRRADIATION over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('IRRADIATION')
plt.show()

print ("minimum="+ str((df_psense1.IRRADIATION).min()) )
print ("maximum="+ str((df_psense1.IRRADIATION).max()) )
print ("mean="+ str((df_psense1.IRRADIATION).mean()) )
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2.DATE_TIME,
        df_psense2.IRRADIATION.rolling(window=40).mean(),
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('IRRADIATION over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('IRRADIATION')
plt.show()

print ("minimum="+ str((df_psense2.IRRADIATION).min()) )
print ("maximum="+ str((df_psense2.IRRADIATION).max()) )
print ("mean="+ str((df_psense2.IRRADIATION).mean()) )
# for plant 1

df_data = df_psense1[df_psense1['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.IRRADIATION.rolling(window=1).mean(),
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Irradiation on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
# for plant 2

df_data = df_psense2[df_psense2['DATE']=='2020-05-23T']

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_data.DATE_TIME,
        df_data.IRRADIATION,
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Variance of Irradiation on 23 May')
plt.xlabel('Date-Time')
plt.ylabel(' Temperature')
plt.show()
daily_yield=df_pgen1.groupby("DATE").agg(TODAY_YIELD=("DAILY_YIELD",max),
                                           DATE=("DATE",max)
                                        )
daily_yield
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(daily_yield.DATE,
        daily_yield.TODAY_YIELD
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DAILY YIELD over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()
print("maximum=" +str( daily_yield['TODAY_YIELD'].max()))
print("minimum=" +str( daily_yield['TODAY_YIELD'].min()))
print("mean=" +str( daily_yield['TODAY_YIELD'].mean()))
daily_yield=df_pgen2.groupby("DATE").agg(TODAY_YIELD=("DAILY_YIELD",max),
                                           DATE=("DATE",max)
                                        )
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(daily_yield.DATE,
        daily_yield.TODAY_YIELD
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DAILY YIELD over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('POWER')
plt.show()
print("maximum=" +str( daily_yield['TODAY_YIELD'].max()))
print("minimum=" +str( daily_yield['TODAY_YIELD'].min()))
print("mean=" +str( daily_yield['TODAY_YIELD'].mean()))
Inverters_performance=df_pgen1.groupby("SOURCE_KEY").agg(LIFETIME_YIELD=("TOTAL_YIELD",max),
                                           SOURCE_KEY=("SOURCE_KEY",max)
                                        )
Inverters_performance
# for plant 1

sns.barplot(x=Inverters_performance["SOURCE_KEY"], y=Inverters_performance["LIFETIME_YIELD"])

print("maximum=" +str(Inverters_performance['LIFETIME_YIELD'].max()))
print("minimum=" +str(Inverters_performance['LIFETIME_YIELD'].min()))
print("mean=" +str( Inverters_performance['LIFETIME_YIELD'].mean()))
Inverters_performance=df_pgen2.groupby("SOURCE_KEY").agg(LIFETIME_YIELD=("TOTAL_YIELD",max),
                                           SOURCE_KEY=("SOURCE_KEY",max)
                                        )
Inverters_performance
#for plant2

sns.barplot(x=Inverters_performance["SOURCE_KEY"], y=Inverters_performance["LIFETIME_YIELD"])

print("maximum=" +str(Inverters_performance['LIFETIME_YIELD'].max()))
print("minimum=" +str(Inverters_performance['LIFETIME_YIELD'].min()))
print("mean=" +str( Inverters_performance['LIFETIME_YIELD'].mean()))
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_psense1.AMBIENT_TEMPERATURE,
        df_psense1.MODULE_TEMPERATURE.rolling(window=5).mean(),
         marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
         )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varying with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


ax.plot(df_psense2.AMBIENT_TEMPERATURE,
        df_psense2.MODULE_TEMPERATURE.rolling(window=5).mean(),
         marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
         )


ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varying with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense1[df_psense1['DATE']=='2020-05-15']

ax.plot(df_data.AMBIENT_TEMPERATURE,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varies with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense2[df_psense2['DATE']=='2020-05-15']

ax.plot(df_data.AMBIENT_TEMPERATURE,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature varies with Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for plant1

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
#for plant2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2['IRRADIATION'],
        df_psense2['MODULE_TEMPERATURE'],
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
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense1[df_psense1['DATE']=='2020-05-23']

ax.plot(df_data.IRRADIATION,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title(' Module Temperature varying with Ambient Temperature on 23rd May')
plt.xlabel('IRRADIATION')
plt.ylabel('Module Temperature')
plt.show()
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))


df_data = df_psense2[df_psense2['DATE']=='2020-05-23']

ax.plot(df_data.IRRADIATION,
        df_data.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title(' Module Temperature varying with Ambient Temperature on 23rd May')
plt.xlabel('IRRADIATION')
plt.ylabel('Module Temperature')
plt.show()
#for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense1['IRRADIATION'],
        df_psense1['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Module Temperature')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperture')
plt.show()
#for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_psense2['IRRADIATION'],
        df_psense2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Module Temperature')
plt.xlabel('Irradiation')
plt.ylabel('Ambient Temperture')
plt.show()
#for plant 1


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.DC_POWER/10,
        df_pgen1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
#for plant 2


_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen2.DC_POWER,
        df_pgen2.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC POWER'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Scatter Plot showing How AC Power varies with DC Power')
plt.xlabel('DC Power')
plt.ylabel('AC Power')
plt.show()
comparision=df_pgen1.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.DC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Daily Yield')
plt.xlabel('DC power')
plt.ylabel('daily yield')
plt.show()
comparision=df_pgen2.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.DC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs Daily Yield')
plt.xlabel('DC power')
plt.ylabel('daily yield')
plt.show()
comparision=df_pgen1.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.AC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Daily Yield')
plt.xlabel('AC power')
plt.ylabel('daily yield')
plt.show()
plt.show()
# for plant 1

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.AC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DAILY YIELD')
plt.xlabel('AC POWER')
plt.ylabel('Daily_Yield')
plt.show()
# for plant 1

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.DC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC POWER and DAILY YIELD')
plt.xlabel('DC POWER')
plt.ylabel('Daily_Yield')
plt.show()
comparision=df_pgen2.groupby("DATE").agg(DAILY_YIELD=("DAILY_YIELD",max),
                                         DC_POWER=("DC_POWER",sum),
                                         AC_POWER=("AC_POWER",sum),
                                         DATE=("DATE",max)
                                         )
comparision
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))



ax.plot(comparision.AC_POWER,
        comparision.DAILY_YIELD,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs Daily Yield')
plt.xlabel('AC power')
plt.ylabel('daily yield')
plt.show()
plt.show()
# for plant 2

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.AC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER and DAILY YIELD')
plt.xlabel('AC POWER')
plt.ylabel('Daily_Yield')
plt.show()
# for plant 2

dates = comparision['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = comparision[comparision['DATE']==date]

    ax.plot(df_data.DC_POWER,
            df_data.DAILY_YIELD,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC POWER and DAILY YIELD')
plt.xlabel('DC POWER')
plt.ylabel('Daily_Yield')
plt.show()
result_outer1 = pd.merge(df_pgen1,df_psense1,on='DATE_TIME',how='outer')
result_outer2 = pd.merge(df_pgen2,df_psense2,on='DATE_TIME',how='outer')
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.IRRADIATION,
        result_outer1.DC_POWER/10,
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
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.IRRADIATION,
        result_outer2.DC_POWER/10,
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
# for plant 1

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
plt.title('AC Power vs. Irradiation')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
# for plant 2

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
plt.title('AC Power vs. Irradiation')
plt.xlabel('Irradiation')
plt.ylabel('AC Power')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.DC_POWER/10,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 2

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
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 1

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer1.MODULE_TEMPERATURE,
        result_outer1.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs. Temperature')
plt.xlabel('Temperature')
plt.ylabel('AC Power')
plt.show()
# for plant 2

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer2.MODULE_TEMPERATURE,
        result_outer2.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='DC POWER')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC Power vs. Temperature')
plt.xlabel('Temperature')
plt.ylabel('AC Power')
plt.show()
# for plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer1[(result_outer1['DATE_x']==date)]

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
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))


for date in dates:
    data = result_outer2[(result_outer2['DATE_x']==date)]

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
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 1

dates = result_outer1['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))



data = result_outer1[(result_outer1['DATE_x']=='2020-05-23')]

ax.plot(data.MODULE_TEMPERATURE,
        data.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
# for plant 2

dates = result_outer2['DATE_x'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))



data = result_outer2[(result_outer2['DATE_x']=='2020-05-23')]

ax.plot(data.MODULE_TEMPERATURE,
        data.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power vs. Module Temperature')
plt.xlabel('Module Temperature')
plt.ylabel('DC Power')
plt.show()
