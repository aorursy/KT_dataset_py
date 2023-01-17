

import pandas as pd

plant1_df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant1_df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant2_df1 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

plant2_df2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
plant1_df1.head() #Generation Data
plant1_df2.head() #Weather Sensor
plant2_df1.head() #Generation Data
plant2_df2.head() #Weather Sensor
plant1_df1.columns
plant1_df2.columns
plant2_df1.columns
plant2_df2.columns
plant1_df1.nunique()
plant1_df2.nunique()
plant2_df1.nunique()
plant2_df2.nunique()
plant1_df1.describe()
plant1_df2.describe()
plant2_df1.describe()
plant2_df2.describe()
mean1 = plant1_df1['DAILY_YIELD'].mean()

print(f'The mean daily yield of plant 1 generation data is {mean1}')
mean2 = plant2_df1['DAILY_YIELD'].mean()

print(f'The mean daily yield of plant 2 generation data is {mean2}')
irradiation_sum1 = plant1_df2['IRRADIATION'].sum()

print(f'The total irradiation in plant1 weather sensor data is {irradiation_sum1}' )
irradiation_sum2 = plant2_df2['IRRADIATION'].sum()

print(f'The total irradiation in plant2 weather sensor data is {irradiation_sum2}' )
max_ambient_1 = plant1_df2['AMBIENT_TEMPERATURE'].max()

max_module_temp_1 = plant1_df2['MODULE_TEMPERATURE'].max()

print(f'The max ambient in plant 1 weather sensor data is {max_ambient_1}')

print(f'The max module temperature in plant 1 weather sensor data is {max_module_temp_1}')
max_ambient_2 = plant2_df2['AMBIENT_TEMPERATURE'].max()

max_module_temp_2 = plant2_df2['MODULE_TEMPERATURE'].max()

print(f'The max ambient in plant 2 weather sensor data is {max_ambient_2}')

print(f'The max module temperature in plant 2 weather sensor data is {max_module_temp_2}')
plant1_inverter_1 = plant1_df1['SOURCE_KEY'].nunique()

plant1_inverter_2 = plant1_df2['SOURCE_KEY'].nunique()

plant2_inverter_1 = plant2_df1['SOURCE_KEY'].nunique()

plant2_inverter_2 = plant2_df2['SOURCE_KEY'].nunique()

print(f'The number of inverters in plant 1 generation data is {plant1_inverter_1}')

print(f'The number of inverters in plant 1 weather sensor data is {plant1_inverter_2}')

print(f'The number of inverters in plant 2 generation data is {plant2_inverter_1}')

print(f'The number of inverters in plant 2 weather sensor data is {plant2_inverter_2}')
print(plant1_df1.info())

print(plant2_df1.info())
#15-05-2020 00:00 ///// 2020-05-15 00:00:00

plant1_df1['DATE_TIME']=pd.to_datetime(plant1_df1['DATE_TIME'],format ='%d-%m-%Y %H:%M')

print(plant1_df1.info())

plant2_df1['DATE_TIME'] = pd.to_datetime(plant2_df1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

print(plant2_df1.info())

plant1_df1['DATE'] = pd.to_datetime(plant1_df1['DATE_TIME'],format ='%d-%m-%Y %H:%M').dt.date

plant1_df1['DATE'] = pd.to_datetime(plant1_df1['DATE'])

plant1_df1
plant1_df2['DATE'] = pd.to_datetime(plant1_df2['DATE_TIME'],format ='%Y-%m-%d %H:%M').dt.date

plant1_df2['DATE'] = pd.to_datetime(plant1_df2['DATE'])

plant1_df2
plant2_df1['DATE'] = pd.to_datetime(plant2_df1['DATE_TIME'],format ='%Y-%m-%d %H:%M:%S').dt.date

plant2_df1['DATE'] = pd.to_datetime(plant2_df1['DATE'])

plant2_df1
plant2_df2['DATE'] = pd.to_datetime(plant2_df2['DATE_TIME'],format ='%Y-%m-%d %H:%M:%S').dt.date

plant2_df2['DATE'] = pd.to_datetime(plant2_df2['DATE'])

d1 = plant1_df1.groupby(['DATE']).sum()

d1
d2=plant2_df1.groupby(['DATE']).sum()

d2
plant1_dc_max_power = (d1['DC_POWER'].max())

plant2_dc_max_power = (d2['DC_POWER'].max())

plant1_dc_min_power = (d1['DC_POWER'].min())

plant2_dc_min_power = (d2['DC_POWER'].min())
print(f'The max dc power for plant 1 in a day is {plant1_dc_max_power}')

print(f'The min dc power for plant 1 in a day is {plant1_dc_min_power}')

print(f'The max dc power for plant 2 in a day is {plant2_dc_max_power}')

print(f'The min dc power for plant 2 in a day is {plant2_dc_min_power}')
plant1_ac_max_power = (d1['AC_POWER'].max())

plant2_ac_max_power = (d2['AC_POWER'].max())

plant1_ac_min_power = (d1['AC_POWER'].min())

plant2_ac_min_power = (d2['AC_POWER'].min())
print(f'The max ac power for plant 1 in a day is {plant1_ac_max_power}')

print(f'The min ac power for plant 1 in a day is {plant1_ac_min_power}')

print(f'The max ac power for plant 2 in a day is {plant2_ac_max_power}')

print(f'The min ac power for plant 2 in a day is {plant2_ac_min_power}')
plant1_df1['DC_POWER'].max()
plant1_df1[plant1_df1['DC_POWER']==plant1_df1['DC_POWER'].max()]['SOURCE_KEY']
plant1_df1['AC_POWER'].max()
plant1_df1[plant1_df1['AC_POWER']==plant1_df1['AC_POWER'].max()]['SOURCE_KEY']
plant2_df1['DC_POWER'].max()
plant2_df1[plant2_df1['DC_POWER']==plant2_df1['DC_POWER'].max()]['SOURCE_KEY']
plant2_df1['AC_POWER'].max()
plant2_df1[plant2_df1['AC_POWER']==plant2_df1['AC_POWER'].max()]['SOURCE_KEY']
plant1_df1["Rank"] = plant1_df1["AC_POWER"].rank(method ='first') 

plant1_df1
plant1_df1["Rank"] = plant1_df1["DC_POWER"].rank(method ='first') 

plant1_df1
plant2_df1["Rank"] = plant2_df1["AC_POWER"].rank(method ='first') 

plant2_df1
plant2_df1["Rank"] = plant2_df1["DC_POWER"].rank(method ='first') 

plant2_df1
print(plant1_df1['SOURCE_KEY'].value_counts())

print(plant1_df1['DATE'].value_counts())
print(plant1_df2['SOURCE_KEY'].value_counts())

print(plant1_df2['DATE'].value_counts())
print(plant2_df1['SOURCE_KEY'].value_counts())

print(plant2_df1['DATE'].value_counts())
plant2_df2['SOURCE_KEY'].value_counts()

plant2_df2['DATE'].value_counts()
plant1_df1.isnull().any().any() # Checks if there is any missing value : False means  no value is missing 





#OTHER WAYS TO CHECK IF THERE IS ANY MISSING

#NUMPY VERSION IS USUALLY FASTER

            # df.isna().any().any()

            # df.isnull().values.any()

            # df.isna().values.any()

            # np.isnan(df.values).any()
plant1_df2.isnull().any().any()
plant2_df1.isnull().any().any()
plant2_df2.isnull().any().any()
plant1_df1.isnull().sum().sum()



#OTHER waYS TO FIND FREQUENCY OF MISSING

            # df.isna().sum().sum()

            # df.isnull().sum().sum()

            # np.isnan(df.values).sum()
plant1_df1.isna().sum()/(len(plant1_df1))*100
temp = plant1_df1.isna().sum()/(len(plant1_df1))*100

print("Column with lowest amount of missings contains {} % missings.".format(temp.min()))

print("Column with highest amount of missings contains {} % missings.".format(temp.max()))
plant1_df1.loc[:, plant1_df1.isnull().any()].columns
plant1_df1.dropna()