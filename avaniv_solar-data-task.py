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



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print('Done')
def read_plant_1_files():

    # path of the CSV file to read

    plant_1_gen_data_filepath = "../input/solar-power-generation-data/Plant_1_Generation_Data.csv"

    plant_1_weather_filepath = "../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv"



    # Read the file into dataframes

    def dateparse (timestamp):

        return pd.to_datetime(timestamp).strftime('%d-%m-%Y %H:%M')



    plant_1_gen_data = pd.read_csv(plant_1_gen_data_filepath, parse_dates=['DATE_TIME'], date_parser=dateparse, index_col = ['DATE_TIME'])

    plant_1_weather_data = pd.read_csv(plant_1_weather_filepath, parse_dates=['DATE_TIME'], index_col = ['DATE_TIME'])

    return plant_1_gen_data, plant_1_weather_data



plant_1_gen_data, plant_1_weather_data = read_plant_1_files()

print(plant_1_gen_data.describe())

print(plant_1_weather_data.describe())
def add_Remove_Cols_prejoin(plant_1_weather_data):

    plant_1_weather_data['Date'] = plant_1_weather_data.index.date #.strftime('%d/%m/%Y')

    plant_1_weather_data['Time'] = plant_1_weather_data.index.time #.strftime('%H:%M')

    # unrequired columns

    plant_1_weather_data.drop(['PLANT_ID', 'SOURCE_KEY'], axis = 1, inplace = True)

    return plant_1_weather_data



plant_1_weather_data = add_Remove_Cols_prejoin(plant_1_weather_data)

plant_1_weather_data['Avg_Irradiation'] = plant_1_weather_data['IRRADIATION'].rolling(2, min_periods=1).mean()



# join the two

print('Joining generation and weather data to create data for plant 1 and adding calculated cols')

plant_1_data = plant_1_gen_data.join(plant_1_weather_data, sort= True)



# adding period yield, yeild/irradiation

def add_calculated_cols(plant_1_data):

    plant_1_data.reset_index(inplace = True)

    plant_1_data = plant_1_data.sort_values(by=['SOURCE_KEY', 'Date', 'Time'])

    plant_1_data['Period_Yield'] = plant_1_data.groupby(['SOURCE_KEY','Date'])['TOTAL_YIELD'].diff().fillna(0)

    plant_1_data['period_yield_dividedby_Irradiation'] = plant_1_data.Period_Yield / plant_1_data.IRRADIATION # check same for with avg irridation too

    plant_1_data['period_yield_dividedby_AvgIrradiation'] = plant_1_data.Period_Yield / plant_1_data.Avg_Irradiation

    return plant_1_data



plant_1_data = add_calculated_cols(plant_1_data)



plant_1_data['Period_length'] = pd.to_datetime(plant_1_data['Time'].astype(str)).diff().dt.total_seconds().div(60*15)

plant_1_data['Period_length'].iloc[0] = 1 # getting warning even if use iloc

plant_1_data['Period_length'] = plant_1_data['Period_length'].where(plant_1_data['Period_length'] > 0, 1) # (24 - plant_1_data['Period_length'].mod(24)))



# adjust period yield/irridation for multiperiod. Note that we only have last period irridation which will be used as proxy for whole multiperiod.

plant_1_data['period_yield_dividedby_Irradiation'] = plant_1_data.period_yield_dividedby_Irradiation / plant_1_data.Period_length

plant_1_data['period_yield_dividedby_AvgIrradiation'] = plant_1_data.period_yield_dividedby_AvgIrradiation / plant_1_data.Period_length

plant_1_data.describe()
plant_1_data_for_charting = plant_1_data[plant_1_data['IRRADIATION'] > 0]



# 1 Daily yield per source key

def draw_Daily_Chart(plant_data):

    Daily_Chart_Data = plant_data.reset_index()

    Daily_Chart_Data = Daily_Chart_Data[['SOURCE_KEY', 'Date', 'DAILY_YIELD']]

    Daily_Chart_Data = Daily_Chart_Data.groupby(['SOURCE_KEY', 'Date']).agg(max)

    print('Chart 1: Daily Yield per Source Key')

    Daily_Chart_Data.unstack(level=0).plot(figsize=(15,10))



    Daily_Chart_Data = plant_data.reset_index()

    Daily_Chart_Data = Daily_Chart_Data[['SOURCE_KEY', 'Date', 'TOTAL_YIELD']]

    Daily_Chart_Data = Daily_Chart_Data.groupby(['SOURCE_KEY', 'Date']).agg(max)

    print('Chart 2: Cumulative Daily Yield per Source Key')

    Daily_Chart_Data.unstack(level=0).plot(figsize=(15,10))



    Daily_Chart_Data = pd.DataFrame()

    

draw_Daily_Chart(plant_1_data)
def draw_Yield_at_Time(data_for_charting):

    YatT_Chart_Data = data_for_charting.reset_index()

    YatT_Chart_Data = YatT_Chart_Data[['SOURCE_KEY', 'Time', 'Period_Yield']]

    # print(YatT_Chart_Data.head(2))

    YatT_Chart_Data = YatT_Chart_Data.groupby(['SOURCE_KEY', 'Time']).agg(sum)

    # print(YatT_Chart_Data.head(2))

    YatT_Chart_Data = YatT_Chart_Data.unstack(level=0)

    #print(YatT_Chart_Data)

    print('Yield at time during the day per Source key')

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.heatmap(YatT_Chart_Data, vmax=10000)

    plt.show()



    YatT_Chart_Data = pd.DataFrame()

    

draw_Yield_at_Time(plant_1_data_for_charting)
def draw_Irradiation_Chart(data_for_charting):

    # see how yield/irradiation is distrubed during the day for each sourcekey

    YbyI_chart_data = data_for_charting.reset_index()



    YbyI_chart_data = YbyI_chart_data[['SOURCE_KEY', 'Time', 'period_yield_dividedby_Irradiation']]



    # YbyI chart

    YbyI_chart_data = YbyI_chart_data.groupby(['SOURCE_KEY', 'Time']).mean()

    YbyI_chart_data = YbyI_chart_data.unstack(level=0)

    # print('Period Yield / Irradiation by Source Key')

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.heatmap(YbyI_chart_data, vmin = 200, vmax = 500)

    plt.show()



    YbyI_chart_data = pd.DataFrame()



def draw_AvgIrradiation_Chart(data_for_charting):

    # see how yield/Avgirradiation is distrubed during the day for each sourcekey

    YbyAvgI_chart_data = data_for_charting.reset_index()



    YbyAvgI_chart_data = YbyAvgI_chart_data[['SOURCE_KEY', 'Time', 'period_yield_dividedby_AvgIrradiation']]



    # YbyI chart

    YbyAvgI_chart_data = YbyAvgI_chart_data.groupby(['SOURCE_KEY', 'Time']).mean()

    YbyAvgI_chart_data = YbyAvgI_chart_data.unstack(level=0)

    # print('Period Yield / Avg Irradiation by Source Key')

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.heatmap(YbyAvgI_chart_data, vmin = 200, vmax = 500)

    plt.show()



    YbyAvgI_chart_data = pd.DataFrame()



def draw_Irradiation_Temp_Charts(data_for_charting):

    Module_Temp_chart_data = data_for_charting.reset_index()

    Ambient_Temp_chart_data = Module_Temp_chart_data

    Module_Temp_chart_data = Module_Temp_chart_data[['SOURCE_KEY', 'MODULE_TEMPERATURE', 'period_yield_dividedby_Irradiation']]

    Ambient_Temp_chart_data = Ambient_Temp_chart_data[['SOURCE_KEY', 'AMBIENT_TEMPERATURE', 'period_yield_dividedby_Irradiation']]



    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=Module_Temp_chart_data['MODULE_TEMPERATURE'], y=Module_Temp_chart_data['period_yield_dividedby_Irradiation'], hue=Module_Temp_chart_data['SOURCE_KEY'])

    # ok, this chart doesnt appear to indicate much (I had expected negative relationship, but dont see it there.)

    plt.ylim(0, 600)

    # Relation to ambient temperature

    # print('Ambient temperature vs Period Yield / Irradiation')

    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=Ambient_Temp_chart_data['AMBIENT_TEMPERATURE'], y=Ambient_Temp_chart_data['period_yield_dividedby_Irradiation'], hue=Ambient_Temp_chart_data['SOURCE_KEY'])

    plt.ylim(0, 600)

    Module_Temp_chart_data = pd.DataFrame()

    Ambient_Temp_chart_data = pd.DataFrame()



def draw_AvgIrradiation_Temp_Charts(data_for_charting):

    Module_Temp_chart_data = data_for_charting.reset_index()

    Ambient_Temp_chart_data = Module_Temp_chart_data

    Module_Temp_chart_data = Module_Temp_chart_data[['SOURCE_KEY', 'MODULE_TEMPERATURE', 'period_yield_dividedby_AvgIrradiation']]

    Ambient_Temp_chart_data = Ambient_Temp_chart_data[['SOURCE_KEY', 'AMBIENT_TEMPERATURE', 'period_yield_dividedby_AvgIrradiation']]



    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=Module_Temp_chart_data['MODULE_TEMPERATURE'], y=Module_Temp_chart_data['period_yield_dividedby_AvgIrradiation'], hue=Module_Temp_chart_data['SOURCE_KEY'])

    # ok, this chart doesnt appear to indicate much (I had expected negative relationship, but dont see it there.)

    plt.ylim(0, 600)

    # Relation to ambient temperature

    # print('Ambient temperature vs Period Yield / Irradiation')

    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=Ambient_Temp_chart_data['AMBIENT_TEMPERATURE'], y=Ambient_Temp_chart_data['period_yield_dividedby_AvgIrradiation'], hue=Ambient_Temp_chart_data['SOURCE_KEY'])

    plt.ylim(0, 600)

    Module_Temp_chart_data = pd.DataFrame()

    Ambient_Temp_chart_data = pd.DataFrame()



draw_Irradiation_Chart(plant_1_data_for_charting)

draw_AvgIrradiation_Chart(plant_1_data_for_charting)
draw_Irradiation_Temp_Charts(plant_1_data_for_charting)

draw_AvgIrradiation_Temp_Charts(plant_1_data_for_charting)
def draw_DC_to_AC_Charts(data_for_charting):

    # AC DC ratios for different source keys

    DC_AC_chart_data = data_for_charting.reset_index()

    DC_AC_chart_data = DC_AC_chart_data[['SOURCE_KEY', 'Time', 'DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]

    DC_AC_chart_data['DC_AC_ratio'] = DC_AC_chart_data['AC_POWER'] / DC_AC_chart_data['DC_POWER']

    DC_AC_ratio_chart_data = DC_AC_chart_data[['SOURCE_KEY', 'Time', 'DC_AC_ratio']]



    # chart

    DC_AC_ratio_chart_data = DC_AC_ratio_chart_data.groupby(['SOURCE_KEY', 'Time']).mean()

    # print(DC_AC_ratio_chart_data)

    DC_AC_ratio_chart_data = DC_AC_ratio_chart_data.unstack(level=0)

    # print(DC_AC_ratio_chart_data)

    # print('DC to AC ratio during the day per source key')

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.heatmap(DC_AC_ratio_chart_data)

    plt.show()



    # did try to plot Dc_AC_ratio to ambient tempurature, but it doesnt seem to be making much sense - deleted



    # DC_AC_ratio to module temp

    Module_Temp_chart_data = DC_AC_chart_data[['SOURCE_KEY', 'MODULE_TEMPERATURE', 'DC_AC_ratio']]

    # print('DC to AC ratio against module temperature')

    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=Module_Temp_chart_data['MODULE_TEMPERATURE'], y=Module_Temp_chart_data['DC_AC_ratio'], hue=Module_Temp_chart_data['SOURCE_KEY'])

    plt.ylim(0.0955, 0.099)



    DC_AC_chart_data = pd.DataFrame()

    DC_AC_ratio_chart_data = pd.DataFrame()

    Module_Temp_chart_data = pd.DataFrame()

    

draw_DC_to_AC_Charts(plant_1_data_for_charting[plant_1_data_for_charting['Period_length']==1]) # removing >1 periods to reduce distortion



def draw_power_to_yield(data_for_charting):

    power_to_yield_chart_data = data_for_charting.reset_index()

    power_to_yield_chart_data = power_to_yield_chart_data[['SOURCE_KEY', 'DATE_TIME', 'AC_POWER', 'Period_Yield']]

    power_to_yield_chart_data['AC_to_Period_Yield_ratio'] = power_to_yield_chart_data['Period_Yield'] / power_to_yield_chart_data['AC_POWER']

    power_to_yield_chart_data = power_to_yield_chart_data[['SOURCE_KEY', 'DATE_TIME', 'AC_to_Period_Yield_ratio']]

    # print('AC Power to Period Yield per Source Key')

    plt.figure(figsize=(15, 10))

    sns.scatterplot(x=power_to_yield_chart_data['DATE_TIME'], y=power_to_yield_chart_data['AC_to_Period_Yield_ratio'], hue=power_to_yield_chart_data['SOURCE_KEY'])

    plt.ylim(0, 1)

    

# why date_time is becoming year starting from some 2004 in below? anyway wasnt much useful, so commenting out for now.

# draw_power_to_yield(plant_1_data_for_charting[plant_1_data_for_charting['Period_length']==1])
# Read files afresh

plant_1_gen_data, plant_1_weather_data = read_plant_1_files()



# count of rows at present

print(plant_1_weather_data.count())



# add missing timestamps to weather data

import datetime

all_timestamps = pd.DataFrame()

all_timestamps['allTimestamps'] = pd.date_range(start=plant_1_weather_data.index[0], end=plant_1_weather_data.index[-1], freq='15min', name='allTimestamps')

all_timestamps = all_timestamps.set_index('allTimestamps')

plant_1_weather_data = plant_1_weather_data.join(all_timestamps, how='outer')



# keep track of which timestamps are added here

plant_1_weather_data['estimated_weather_vals'] = plant_1_weather_data.PLANT_ID.isnull()



# add cols Date, Time (and remove unnecessory ones - PLANT_ID, SOURCE_KEY )

plant_1_weather_data = add_Remove_Cols_prejoin(plant_1_weather_data)



# print(plant_1_weather_data.head(2))



# populate missing values.

# I am using prev n days time-specific averages as representative values. -this is NOT the best approach (better adjust

# for avg departure of current day too). prob. in next iteration.



plant_1_weather_data.sort_index()

cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

for col in cols:

    plant_1_weather_data[col] = plant_1_weather_data.groupby('Time')[col].transform(lambda x: x.fillna(x.rolling(3, 1).mean()))

    plant_1_weather_data[col] = plant_1_weather_data.groupby('Time')[col].transform(lambda x: x.fillna(x.mean())) # this if some had no rolling values

plant_1_weather_data['Avg_Irradiation'] = plant_1_weather_data['IRRADIATION'].rolling(2, min_periods=1).mean()

plant_1_weather_data['Cumulative_Irradiation'] = plant_1_weather_data.groupby('Date')['IRRADIATION'].cumsum() # for generation data interpolation

print(plant_1_weather_data.describe())



# Join generation data and weather data

# outer join between the two doesnt work because some 100+ timestamps dont have corrosponding data and

# we wont know which source keys are missing which timestamps.

# we will have to join therefore by each source key



plant_1_data = pd.DataFrame()

plantid = plant_1_gen_data['PLANT_ID'][0]

for sk in plant_1_gen_data.SOURCE_KEY.unique():

    sk_data =  pd.DataFrame()

    sk_data = plant_1_gen_data[plant_1_gen_data['SOURCE_KEY'] == sk]

    sk_data = plant_1_weather_data.join(sk_data, how='left')

    sk_data['estimated_gen_vals'] = sk_data.PLANT_ID.isnull()

    sk_data[['PLANT_ID', 'SOURCE_KEY']] = sk_data[['PLANT_ID', 'SOURCE_KEY']].fillna(value = {'PLANT_ID': plantid, 'SOURCE_KEY': sk}) # remove plantid value

    sk_data.index.name = 'DATE_TIME'

    sk_data.sort_index(axis = 0, inplace= True)

    # because we dont want values allocated where irradiation is 0 or want interpolation to be based on points on them,

    # nor do we have any use of points with 0 irradiation, we can delete those points before interpolation.

    # before deleting however, make sure yield for the day is reflacted in last total yield (case where data collection has faied around eod)

    get_eod_ty = sk_data.groupby('Date')['TOTAL_YIELD'].max() # max yield for the day

    sk_data = sk_data[sk_data['IRRADIATION'] > 0]

    get_eod_times = sk_data.groupby('Date')['Time'].max() # max time when irridation was > 0

    eod_ty = pd.merge(get_eod_ty, get_eod_times, right_index = True, left_index = True)

    for row in eod_ty.itertuples():

        # put total yield at eod - whether or not it is na

        mask = (sk_data['Date']==row[0]) & (sk_data['Time']==row[2])

        sk_data.loc[mask, 'TOTAL_YIELD'] = row[1] #fillna is better chage back to it sk_data.loc[mask, 'TOTAL_YIELD'].fillna(row[1])

    # power should be interpolated based on irradiation, currently its liner

    sk_data['DC_POWER'].interpolate(inplace = True)

    sk_data['AC_POWER'].interpolate(inplace = True)

    # Interpolate values based on cum_Irradiation

    sk_data = sk_data.reset_index()

    sk_data = sk_data.set_index(['Cumulative_Irradiation'], drop=True)

    # in below, apply gives key error: 0 so transform - this also drops column 'Date' - added later manually

    sk_data = sk_data.groupby('Date').transform(lambda g: g.interpolate(method = 'index', limit_direction='both'))

    # get back to date_time index

    sk_data.reset_index(inplace = True)

    sk_data.set_index(['DATE_TIME'], drop=True, inplace = True)

    # this is workaround of date being dropped in above groupby. remove this once figure out how to avoid it getting lost

    sk_data['Date'] = sk_data.index.date

    sk_data.sort_index(axis = 0, inplace = True)

    # fill period yield and daily yield na values

    sk_data['Period_Yield'] = sk_data['TOTAL_YIELD'].diff().fillna(0) # daily yield midnight data has strange value. its not so for total yield

    sk_data['Cum_Period_Yield'] = sk_data.groupby('Date')['Period_Yield'].cumsum()

    # Note:some existing values for daily yield are wrong. e.g. 21-05-2020 08:00, 4135001, 1BY6WEcLGh8j5v7, 3089.833333, 303.1166667, 241.3333333, 6298697

    # these existing incorrect values are not touched as of now. -though Cum_Period_Yield can serve as Daily yield

    sk_data['DAILY_YIELD'] = sk_data['DAILY_YIELD'].transform(lambda x: x.fillna(sk_data['Cum_Period_Yield']))

    sk_data['period_yield_dividedby_Irradiation'] = sk_data.Period_Yield / sk_data.IRRADIATION

    sk_data['period_yield_dividedby_AvgIrradiation'] = sk_data.Period_Yield / sk_data.Avg_Irradiation

    plant_1_data = pd.concat([plant_1_data, sk_data])

    # - this isnt working: plant_1_data = plant_1_data.round(2)

print(plant_1_data.describe())

Interpolated_data_for_charting = plant_1_data



# draw_Daily_Chart(plant_1_data)



draw_Irradiation_Chart(Interpolated_data_for_charting)

draw_AvgIrradiation_Chart(Interpolated_data_for_charting)

draw_Yield_at_Time(Interpolated_data_for_charting)

draw_DC_to_AC_Charts(Interpolated_data_for_charting)

# draw_power_to_yield(Interpolated_data_for_charting)



Interpolated_data_for_charting = pd.DataFrame()