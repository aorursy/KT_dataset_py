import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# reading csv dataset file

plant1_gen = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant1_gen.name = 'Plant 1 Generation'

plant1_weth = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant1_weth.name = 'Plant 1 weather sensor'

plant2_gen = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

plant2_gen.name = 'Plant 2 generation'

plant2_weth = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

plant2_weth.name = 'Plant 2 weather sensor'



plant_gen = [plant1_gen,plant2_gen]

plant_weth = [plant1_weth,plant2_weth]
for plant in plant_gen:

    print('The basic stats of {} are \n{} \n'.format(plant.name,plant.describe()))

    print('Columns in {}'.format(plant.name))

    x = plant.columns

    for j in x:

        print(j+'\n')

    

for plant in plant_weth:

    print('The basic stats of {} are \n{} \n'.format(plant.name,plant.describe()))

    print('Columns in {}'.format(plant.name))

    x = plant.columns

    for j in x:

        print(j+'\n')
for plant in plant_gen:

    print("The mean value of daily yield in {} is {}".format(plant.name,plant['DAILY_YIELD'].mean()))

for plant in plant_weth:

    plant1_weth['DATES'] = pd.to_datetime(plant1_weth['DATE_TIME']).dt.date

    gruped = plant1_weth.groupby('DATES')

    print("The total irradiation per day for {} is \n {} \n".format(plant.name,gruped['IRRADIATION'].sum()))

for plant in plant_weth:

    max_ambtemp = plant['AMBIENT_TEMPERATURE'].max()

    max_modtemp = plant['MODULE_TEMPERATURE'].max()

    min_ambtemp = plant['AMBIENT_TEMPERATURE'].min()

    min_modtemp = plant['MODULE_TEMPERATURE'].min()

    print("The maximum ambient and module temperature of {} are {} and {} respectively.".format(plant.name,max_ambtemp,max_modtemp))

    print("The minimum ambient and module temperature of {} are {} and {} respectively.".format(plant.name,min_ambtemp,min_modtemp))
for plant in plant_gen:

    inverters = plant['SOURCE_KEY'].nunique()

    print("Number of iverters in {} is {}".format(plant.name,inverters))
for plant in plant_gen:

    plant1_gen['DATES'] = pd.to_datetime(plant1_gen['DATE_TIME']).dt.date

    gruped = plant1_gen.groupby('DATES')

    print("#####The maximum AC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['AC_POWER'].max().reset_index()))

    print("#####The minimum AC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['AC_POWER'].min().reset_index()))
for plant in plant_gen:

    plant1_gen['DATES'] = pd.to_datetime(plant1_gen['DATE_TIME']).dt.date

    gruped = plant1_gen.groupby('DATES')

    print("#####The maximum DC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['DC_POWER'].max().reset_index()))

    print("#####The minimum DC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['DC_POWER'].min().reset_index()))
for plant in plant_gen:

    plant1_gen['TIME'] = pd.to_datetime(plant1_gen['DATE_TIME']).dt.hour

    gruped = plant1_gen.groupby('TIME')

    print("#####The maximum DC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['DC_POWER'].max().reset_index()))

    print("#####The minimum DC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['DC_POWER'].min().reset_index()))
for plant in plant_gen:

    plant1_gen['TIME'] = pd.to_datetime(plant1_gen['DATE_TIME']).dt.hour

    gruped = plant1_gen.groupby('TIME')

    print("#####The maximum AC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['AC_POWER'].max().reset_index()))

    print("#####The minimum AC power generated per day for {} is \n {}\n\n".format(plant.name,gruped['AC_POWER'].min().reset_index()))
for plant in plant_gen:

    inverter_maxAC = plant.iloc[plant['AC_POWER'].argmax()]['SOURCE_KEY']

    print("The source key of the inverter producing max AC power in {} is {}".format(plant.name,inverter_maxAC))
for plant in plant_gen:

    ranked = plant.sort_values('AC_POWER',ascending=False)

    print("Ranked list of {} is \n {}\n".format(plant.name,ranked['AC_POWER'].reset_index()))

    print("Top ranked inverter is {}\n\n".format(ranked.iloc[ranked['AC_POWER'].argmax()]['SOURCE_KEY']))
for plant in plant_gen:

    ranked = plant.sort_values('DC_POWER',ascending=False)

    print("Ranked list of {} is \n {}\n".format(plant.name,ranked['DC_POWER'].reset_index()))

    print("Top ranked inverter is {}\n\n".format(ranked.iloc[ranked['DC_POWER'].argmax()]['SOURCE_KEY']))
for plant in plant_gen:

    print("The total number of null columns present in {}\n".format(plant.name))

    print('{}\n'.format(plant.isnull().sum()))

    null_cols = plant.isnull().sum().sum()

    if not null_cols:

        print("Hence, there are no missing datas in {}\n\n".format(plant.name))

    else:

        print("There are {} missing datas in {}\n\n".format(null_cols,plant.name))