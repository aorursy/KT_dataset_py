

import pandas as pd

df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df1.head() # Displays Dataset of Generation data
df2.head() # Dataset of Weather Sensor Data
# COnvert the DATE_TIME column from String datatype to Datetime datatype

df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

# Splitting DateTime column into Date column and Time column

df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date

df1['DATE'] = pd.to_datetime(df1['DATE'])

df1.info()
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M')

df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date

df2['DATE'] = pd.to_datetime(df2['DATE'])

df2.info()
df1.columns
df2.columns
df1.nunique()
df2.nunique()
df1.describe()
df2.describe()
dy_mean = df1['DAILY_YIELD'].mean()
print(f'The mean value of Daily Yield is {dy_mean}')
df2['IRRADIATION'].sum()
df2.groupby('DATE')["IRRADIATION"].sum()

df2['AMBIENT_TEMPERATURE'].max()
df2['MODULE_TEMPERATURE'].max()
df1['SOURCE_KEY'].nunique()
df1['DC_POWER'].max()
df1['DC_POWER'].min()
df1['AC_POWER'].max()
df1['AC_POWER'].min()
#plant1 min dc_power

a=df1.groupby('DATE')["DC_POWER"].min()

print (a)
#plant1 max dc_power

b=df1.groupby('DATE').max()['DC_POWER']

b
df1[df1['DC_POWER']==df1['DC_POWER'].max()]['SOURCE_KEY']

# Take the max value of DC Power column and check it in a DataFrame column of DC Power
df1['DC_POWER']
df1['DC_POWER'].max()
df1['DC_POWER']==df1['DC_POWER'].max()
df1[df1['DC_POWER']==df1['DC_POWER'].max()]['SOURCE_KEY']
#forplant1

s = df1.groupby('SOURCE_KEY').sum()

s["AC_POWER"].sort_values()
#forplant1

s = df1.groupby('SOURCE_KEY').sum()

s["DC_POWER"].sort_values()