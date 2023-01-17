# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime



from bq_helper import BigQueryHelper



from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

from matplotlib import dates as md

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_rawdata = '/kaggle/input/cubems-smart-building-energy-and-iaq-data/'
list_csv = []



for file in os.listdir(path_rawdata):

    if file.endswith(".csv"):

        list_csv.append(os.path.join(path_rawdata, file))

        

list_csv
df_merged = pd.DataFrame({'Date': pd.date_range('2018-07-01', '2020-01-01', freq='min', closed='left')}).set_index('Date')



for path_csv in list_csv:

    csv_file_name = path_csv.split('/')[-1]

    print(csv_file_name)

    df_temp = pd.read_csv(path_csv)

    df_temp = df_temp.set_index('Date')

    df_temp = df_temp.dropna(how='all')

    df_temp.index = pd.to_datetime(df_temp.index)





    str_floor = pd.Series(csv_file_name).str.split('2018|2019|.csv',expand=True).replace('', np.nan).dropna(axis=1).iloc[0,0]

    df_temp.columns = str_floor + '_' + df_temp.columns

    

    df_merged.loc[df_temp.index, df_temp.columns] = df_temp
df_merged.iloc[:, :5].resample('H').mean().iplot()
df_merged.sort_index(axis=0).sort_index(axis=1).to_pickle('df_merged.pickle.gz', compression='gzip')
# Input parameters

station_name = 'BANGKOK METROPOLIS'

years = range(2018, 2020)
helper = BigQueryHelper('bigquery-public-data', 'noaa_gsod')



sql = '''

SELECT

  year, mo, da, temp, min, max, prcp

FROM

    `bigquery-public-data.noaa_gsod.gsod{}` a

    INNER JOIN `bigquery-public-data.noaa_gsod.stations` b ON a.stn = b.usaf

WHERE

  country = 'TH' AND name = '{}'

'''



# Query weather data for each year

datasets = [ helper.query_to_pandas(sql.format(i, station_name)) for i in years ]



# print out each year's data shape

print('\n'.join([ '{}: {}'.format(x[0],x[1].shape) for x in zip(years, datasets)]))
# Concatenate datasets

weather = pd.concat(datasets)



# Handling missing values based on Table Schema description

weather['temp'] = weather['temp'].replace({ 9999.9 : np.nan })

weather['min'] = weather['min'].replace({ 9999.9 : np.nan })

weather['max'] = weather['max'].replace({ 9999.9 : np.nan })

weather['prcp'] = weather['prcp'].replace( { 99.99 : 0 })



weather
# Data processing



# Setting date index

weather['date'] = weather.apply(lambda x: 

                                datetime.datetime(int(x.year), int(x.mo), int(x.da)), 

                                axis=1)

weather = weather.set_index('date')



# Convert temperature values from farenheit to celcius

def f_to_c(temp_f):

    temp_c = (temp_f - 32) * 5/9

    return round(temp_c, 2)



for col in ['temp','min','max']:

    weather[col] = weather[col].apply(f_to_c)



# Convert precipitation from inches to milimeters

def inch_to_mm(x):

    return round(x * 25.4)



weather['prcp'] = weather['prcp'].apply(inch_to_mm)
start_date = '{}0101'.format(years[0])

end_date = weather.index.max().strftime('%Y%m%d')



# Re-index to see if there are any days with missing data

weather = weather.reindex(pd.date_range(start_date, end_date))



# check if there is missing values occured from re-indexing

missing = weather[weather.isnull().any(axis=1)].index

# printing missing index

missing
# Interpolate numerical variables for the missing days

weather = weather.interpolate()



# Re-setting year, month, day fields

weather['year'] = weather.index.year

weather['mo'] = weather.index.month

weather['da'] = weather.index.day



# Verify interpolated data

weather.loc[missing].head(10)
data = weather[['temp','min','max','prcp']]

data.columns = ['Avg Temp', 'Min Temp', 'Max Temp', 'Precip']
data.iplot()
data.to_pickle('df_weather.pickle.gz', compression='gzip')
df_holiday_2018 = pd.read_html('https://www.timeanddate.com/holidays/thailand/2018')[0]

df_holiday_2018.columns = df_holiday_2018.columns.get_level_values(0)

df_holiday_2018 = df_holiday_2018.dropna(how='all')

df_holiday_2018 = df_holiday_2018[['Date', 'Name', 'Type']]

df_holiday_2018['Date'] = '2018 ' + df_holiday_2018['Date']

df_holiday_2018['Date'] = pd.to_datetime(df_holiday_2018['Date'])



df_holiday_2019 = pd.read_html('https://www.timeanddate.com/holidays/thailand/2019')[0]

df_holiday_2019.columns = df_holiday_2019.columns.get_level_values(0)

df_holiday_2019 = df_holiday_2019.dropna(how='all')

df_holiday_2019 = df_holiday_2019[['Date', 'Name', 'Type']]

df_holiday_2019['Date'] = '2019 ' + df_holiday_2019['Date']

df_holiday_2019['Date'] = pd.to_datetime(df_holiday_2019['Date'])



df_holiday = pd.concat([df_holiday_2018, df_holiday_2019], axis=0, ignore_index=True)

df_holiday = df_holiday.drop_duplicates(subset=['Date'])

df_holiday = df_holiday.set_index('Date').asfreq('D')

df_holiday.loc[df_holiday.index.weekday>=5, 'Name'] = 'weekend'

df_holiday.loc[df_holiday.index.weekday>=5, 'Type'] = 'weekend'

df_holiday.columns = 'holiday_' + df_holiday.columns



df_holiday = df_holiday.reset_index()

df_holiday = df_holiday.rename(columns={'Date':'date'}) 



df_holiday
df_holiday_encode = df_holiday.copy()

df_holiday_encode[['holiday_Name', 'holiday_Type']] = df_holiday_encode[['holiday_Name', 'holiday_Type']].astype('str').apply(LabelEncoder().fit_transform)

df_holiday_encode
df_holiday_encode.set_index('date')['holiday_Type'].iplot()
data.to_pickle('df_holiday.pickle.gz', compression='gzip')

data.to_pickle('df_holiday_encode.pickle.gz', compression='gzip')