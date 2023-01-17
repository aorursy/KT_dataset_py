import pandas as pd
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
plant2_df1 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
plant2_df2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')



df1.head()#plant 1
plant2_df1.head()#plant 2
df2.head()
plant2_df2.head()
df1.columns
df2.columns
plant2_df1.columns
plant2_df2.columns
df1.nunique()
df2.nunique()
plant2_df1.nunique()
plant2_df2.nunique()
df1.describe()
df2.describe()
plant2_df1.describe()
plant2_df2.describe()
dy_mean = df1['DAILY_YIELD'].mean()
print(f'The mean value of Daily Yield is {dy_mean}')
dy_mean2 = plant2_df1['DAILY_YIELD'].mean()
print(f'The mean daily yield of plant 2 generation data is {dy_mean2}')
irradiation_sum1 = df2['IRRADIATION'].sum()
print(f'The total irradiation in plant1 weather sensor data is {irradiation_sum1}' )
irradiation_sum2 = plant2_df2['IRRADIATION'].sum()
print(f'The total irradiation in plant2 weather sensor data is {irradiation_sum2}' )

max_ambient_1 = df2['AMBIENT_TEMPERATURE'].max()
max_module_temp_1 = df2['MODULE_TEMPERATURE'].max()
print(f'The max ambient in plant 1 weather sensor data is {max_ambient_1}')
print(f'The max module temperature in plant 1 weather sensor data is {max_module_temp_1}')
max_ambient_2 = plant2_df2['AMBIENT_TEMPERATURE'].max()
max_module_temp_2 = plant2_df2['MODULE_TEMPERATURE'].max()
print(f'The max ambient in plant 2 weather sensor data is {max_ambient_2}')
print(f'The max module temperature in plant 2 weather sensor data is {max_module_temp_2}')
plant1_inverter1 = df1['SOURCE_KEY'].nunique()
plant1_inverter2 = df2['SOURCE_KEY'].nunique()
plant2_inverter1 = plant2_df1['SOURCE_KEY'].nunique()
plant2_inverter2 = plant2_df2['SOURCE_KEY'].nunique()
print(f'The number of inverters in plant 1 generation data is {plant1_inverter1}')
print(f'The number of inverters in plant 1 weather sensor data is {plant1_inverter2}')
print(f'The number of inverters in plant 2 generation data is {plant2_inverter1}')
print(f'The number of inverters in plant 2 weather sensor data is {plant2_inverter2}')

df1['DC_POWER'].max()

df1['DC_POWER'].min()
print(df1.info())
print(plant2_df1.info())

df1['DATE_TIME']=pd.to_datetime(df1['DATE_TIME'],format ='%d-%m-%Y %H:%M')
print(df1.info())
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
print(df1.info())
df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format ='%d-%m-%Y %H:%M').dt.date
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1
df2['DATE'] = pd.to_datetime(df2['DATE_TIME'],format ='%Y-%m-%d %H:%M:%S').dt.date
df2['DATE'] = pd.to_datetime(df2['DATE'])
plant2_df1['DATE'] = pd.to_datetime(plant2_df1['DATE_TIME'],format ='%Y-%m-%d %H:%M:%S').dt.date
plant2_df1['DATE'] = pd.to_datetime(plant2_df1['DATE'])
plant2_df1
d1 = df1.groupby(['DATE']).sum()
d1
plant2_df2['DATE'] = pd.to_datetime(plant2_df2['DATE_TIME'],format ='%Y-%m-%d %H:%M:%S').dt.date
plant2_df2['DATE'] = pd.to_datetime(plant2_df2['DATE'])
d1 = df1.groupby(['DATE']).sum()
d1
d2=df1.groupby(['DATE']).sum()
d2
plant1_dc_max_power = (d1['DC_POWER'].max())
plant2_dc_max_power = (d2['DC_POWER'].max())
plant1_dc_min_power = (d1['DC_POWER'].min())
plant2_dc_min_power = (d2['DC_POWER'].min())
print(f'The max dc power for plant 1 one table in a day is {plant1_dc_max_power}')
print(f'The min dc power for plant 1 other table in a day is {plant1_dc_min_power}')
print(f'The max dc power for plant 2 first table in a day is {plant2_dc_max_power}')
print(f'The min dc power for plant 2 second table in a day is {plant2_dc_min_power}')
plant1_ac_max_power = (d1['AC_POWER'].max())
plant2_ac_max_power = (d2['AC_POWER'].max())
plant1_ac_min_power = (d1['AC_POWER'].min())
plant2_ac_min_power = (d2['AC_POWER'].min())
print(f'The max ac power for plant 1 table 1 in a day is {plant1_ac_max_power}')
print(f'The min ac power for plant 1 table 2 in a day is {plant1_ac_min_power}')
print(f'The max ac power for plant 2 table 1 in a day is {plant2_ac_max_power}')
print(f'The min ac power for plant 2 table 2 in a day is {plant2_ac_min_power}')
df1['DC_POWER']
df1['DC_POWER'].max()
df1['DC_POWER']==df1['DC_POWER'].max()
df1[df1['DC_POWER']==df1['DC_POWER'].max()]['SOURCE_KEY']
plant2_df1['DC_POWER'].max()
plant2_df1[plant2_df1['DC_POWER']==plant2_df1['DC_POWER'].max()]['SOURCE_KEY']
df1['AC_POWER']
df1['AC_POWER'].max()
df1['AC_POWER']==df1['AC_POWER'].max()
df1[df1['AC_POWER']==df1['AC_POWER'].max()]['SOURCE_KEY']
plant2_df1['AC_POWER'].max()
plant2_df1[plant2_df1['AC_POWER']==plant2_df1['AC_POWER'].max()]['SOURCE_KEY']
#all invertors are ranked individually
df1["Rank"] = df1["AC_POWER"].rank(method ='first') 
df1
df1["Rank"] = df1["DC_POWER"].rank(method ='first') 
df1
plant2_df1["Rank"] = plant2_df1["AC_POWER"].rank(method ='first') 
plant2_df1
plant2_df1["Rank"] = plant2_df1["DC_POWER"].rank(method ='first') 
plant2_df1
print(df1['SOURCE_KEY'].value_counts())
print(df1['DATE'].value_counts())

print(df2['SOURCE_KEY'].value_counts())
print(df2['DATE'].value_counts())
print(plant2_df1['SOURCE_KEY'].value_counts())
print(plant2_df1['DATE'].value_counts())
plant2_df2['SOURCE_KEY'].value_counts()
plant2_df2['DATE'].value_counts()
df1.isnull().any().any()
df2.isnull().any().any()
plant2_df1.isnull().any().any()
plant2_df2.isnull().any().any()
df1.loc[:, df1.isnull().any()].columns
df1.dropna()
