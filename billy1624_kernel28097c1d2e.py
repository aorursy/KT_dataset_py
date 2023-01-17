import pandas as pd

from multiprocessing import Pool

import numpy as np

import multiprocessing as mp

import time



def parallelize_dataframe(df, func):

    num_cores = mp.cpu_count()

    df_split = np.array_split(df, num_cores)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



pd.options.display.max_rows

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', -1)
# https://www.kaggle.com/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018

a_fd = pd.read_csv('../input/airline-delay-and-cancellation-data-2009-2018/2015.csv')

# a_fd = pd.read_csv('/content/drive/My Drive/_COMP4462/2015_a.csv')
a_with_delay = a_fd[(a_fd['ARR_DELAY'] > 0.0) & (pd.notnull(a_fd['CARRIER_DELAY']))]

a_with_delay.head()
a_with_delay.shape
a_no_delay = a_fd[(a_fd['ARR_DELAY'] <= 0.0)]

a_no_delay.head()
a_no_delay.shape
print('{:,}'.format(a_fd.shape[0] - a_with_delay.shape[0] - a_no_delay.shape[0]))
a_with_delay.groupby(['DEST']).groups.keys()
a_annual_flight = a_fd.copy()



a_annual_flight['num'] = 1



a_annual_flight_dest = a_annual_flight.groupby(['DEST'], as_index=False)['num'].sum()

a_annual_flight_dest.columns = ['code', 'annual_dest']

a_annual_flight_origin = a_annual_flight.groupby(['ORIGIN'], as_index=False)['num'].sum()

a_annual_flight_origin.columns = ['code', 'annual_origin']



a_annual_flight_grouped = pd.merge(a_annual_flight_dest, a_annual_flight_origin, on='code', how='outer')



del a_annual_flight

del a_annual_flight_dest

del a_annual_flight_origin



a_annual_flight_grouped
a_average_flight_delay = a_fd.copy()



a_average_flight_delay['num'] = 1



a_average_flight_delay_dest = a_average_flight_delay.groupby(['DEST'], as_index=False)['num', 'ARR_DELAY'].sum()

a_average_flight_delay_dest.columns = ['code', 'num_arr', 'average_arr_delay']

a_average_flight_delay_origin = a_average_flight_delay.groupby(['ORIGIN'], as_index=False)['num', 'DEP_DELAY'].sum()

a_average_flight_delay_origin.columns = ['code', 'num_dep', 'average_dep_delay']



a_average_flight_delay_grouped = pd.merge(a_average_flight_delay_dest, a_average_flight_delay_origin, on='code', how='outer')



def calculate_average(row):

    row['average_arr_delay'] = round(row['average_arr_delay'] / row['num_arr'], 2)

    row['average_dep_delay'] = round(row['average_dep_delay'] / row['num_dep'], 2)

    return row

def df_calculate_average(df):

    return df.apply(calculate_average, axis=1)



a_average_flight_delay_grouped = parallelize_dataframe(a_average_flight_delay_grouped, df_calculate_average)

a_average_flight_delay_grouped = a_average_flight_delay_grouped.drop(columns=['num_arr', 'num_dep'])



del a_average_flight_delay

del a_average_flight_delay_dest

del a_average_flight_delay_origin



a_average_flight_delay_grouped
a_30mins_flight_delay = a_fd.copy()



a_30mins_flight_delay['num'] = 1

a_30mins_flight_delay['30mins_arr_delay'] = 0

a_30mins_flight_delay['30mins_dep_delay'] = 0



def calculate_30min_delay(delay):

    if delay >= 30.0:

        return 1

    return 0

def df_arr_calculate_30min_delay(df):

    df['30mins_arr_delay'] = df['ARR_DELAY'].apply(calculate_30min_delay)

    return df

def df_dep_calculate_30min_delay(df):

    df['30mins_dep_delay'] = df['DEP_DELAY'].apply(calculate_30min_delay)

    return df



a_30mins_flight_delay = parallelize_dataframe(a_30mins_flight_delay, df_arr_calculate_30min_delay)

a_30mins_flight_delay = parallelize_dataframe(a_30mins_flight_delay, df_dep_calculate_30min_delay)
a_30mins_flight_delay[a_30mins_flight_delay['30mins_dep_delay'] == 1].head()
a_30mins_flight_delay_dest = a_30mins_flight_delay.groupby(['DEST'], as_index=False)['num', '30mins_arr_delay'].sum()

a_30mins_flight_delay_dest.columns = ['code', 'num', '30mins_delay']

a_30mins_flight_delay_origin = a_30mins_flight_delay.groupby(['ORIGIN'], as_index=False)['num', '30mins_dep_delay'].sum()

a_30mins_flight_delay_origin.columns = ['code', 'num', '30mins_delay']



def calculate_average(row):

    row['30mins_delay'] = round(row['30mins_delay'] / row['num'] * 100, 2)

    return row

def df_calculate_average(df):

    return df.apply(calculate_average, axis=1)



a_30mins_flight_delay_dest = parallelize_dataframe(a_30mins_flight_delay_dest, df_calculate_average)

a_30mins_flight_delay_origin = parallelize_dataframe(a_30mins_flight_delay_origin, df_calculate_average)



a_30mins_flight_delay_dest.columns = ['code', 'num_arr', '30mins_arr_delay']

a_30mins_flight_delay_origin.columns = ['code', 'num_dep', '30mins_dep_delay']



a_30mins_flight_delay_grouped = pd.merge(a_30mins_flight_delay_dest, a_30mins_flight_delay_origin, on='code', how='outer')

a_30mins_flight_delay_grouped = a_30mins_flight_delay_grouped.drop(columns=['num_arr', 'num_dep'])



del a_30mins_flight_delay

del a_30mins_flight_delay_dest

del a_30mins_flight_delay_origin



a_30mins_flight_delay_grouped
a_delay_cause = a_fd.copy()



a_delay_cause['CARRIER_DELAY'].fillna(0, inplace=True)

a_delay_cause['WEATHER_DELAY'].fillna(0, inplace=True)

a_delay_cause['NAS_DELAY'].fillna(0, inplace=True)

a_delay_cause['SECURITY_DELAY'].fillna(0, inplace=True)

a_delay_cause['LATE_AIRCRAFT_DELAY'].fillna(0, inplace=True)



a_delay_cause = a_delay_cause.groupby(['DEST'], as_index=False)['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'].sum()



def calculate_average(row):

    total = row['CARRIER_DELAY']+row['WEATHER_DELAY']+row['NAS_DELAY']+row['SECURITY_DELAY']+row['LATE_AIRCRAFT_DELAY']

    row['CARRIER_DELAY'] = round(row['CARRIER_DELAY'] / total * 100, 2)

    row['WEATHER_DELAY'] = round(row['WEATHER_DELAY'] / total * 100, 2)

    row['NAS_DELAY'] = round(row['NAS_DELAY'] / total * 100, 2)

    row['SECURITY_DELAY'] = round(row['SECURITY_DELAY'] / total * 100, 2)

    row['LATE_AIRCRAFT_DELAY'] = round(row['LATE_AIRCRAFT_DELAY'] / total * 100, 2)

    return row



a_delay_cause = a_delay_cause.apply(calculate_average, axis=1)

a_delay_cause.columns = ['code', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

a_delay_cause
a_cancellation_rate = a_fd.copy()



a_cancellation_rate['num'] = 1



a_cancellation_rate = a_cancellation_rate.groupby(['DEST'], as_index=False)['num', 'CANCELLED'].sum()



def calculate_average(row):

    row['CANCELLED'] = round(row['CANCELLED'] / row['num'] * 100, 2)

    return row



a_cancellation_rate = a_cancellation_rate.apply(calculate_average, axis=1)

a_cancellation_rate = a_cancellation_rate.drop(columns=['num'])

a_cancellation_rate.columns = ['code', 'cancellation_rate']

a_cancellation_rate
# a_annual_flight_grouped

# a_average_flight_delay_grouped

# a_30mins_flight_delay_grouped

# a_delay_cause

# a_cancellation_rate



final_grouped = pd.merge(a_annual_flight_grouped, a_average_flight_delay_grouped, on='code', how='outer')

final_grouped = pd.merge(final_grouped, a_30mins_flight_delay_grouped, on='code', how='outer')

final_grouped = pd.merge(final_grouped, a_delay_cause, on='code', how='outer')

final_grouped = pd.merge(final_grouped, a_cancellation_rate, on='code', how='outer')



a_delay_cause.columns = list(map(lambda x: str(x).lower(), a_delay_cause.columns))



final_grouped.head()
us_airports_df = pd.read_csv('../input/us-airports/us_airports.csv', encoding='unicode_escape')

us_airports_df.head()
us_airports_df = us_airports_df.drop(columns=['type', 'home_link', 'wikipedia_link'])

us_airports_df.columns = ['code', 'name', 'lat', 'lon', 'elevation_ft', 'iso_region', 'municipality', 'icao_code']

us_airports_df.head()
final_grouped_with_airport_info = pd.merge(final_grouped, us_airports_df, on='code', how='left')

without_airport_info = pd.isnull(final_grouped_with_airport_info['lat'])

final_grouped_with_airport_info[without_airport_info]
final_grouped_with_airport_info = final_grouped_with_airport_info[~without_airport_info]

final_grouped_with_airport_info.head()
final_grouped_with_airport_info['total_delay'] = final_grouped_with_airport_info['30mins_arr_delay'] + final_grouped_with_airport_info['30mins_dep_delay']

final_grouped_with_airport_info.head()
final_grouped_with_airport_info = final_grouped_with_airport_info.sort_values(by=['total_delay'], ascending=True)



final_grouped_with_airport_info['efficiency_score'] = 0.0



i = 100

step = i / final_grouped_with_airport_info.shape[0]



# print(final_grouped_with_airport_info.shape)

# print(i, step)



def calculate_average(total_delay):

    global i

    global step

    score = i

    i = i - step

    return round(score, 2)



final_grouped_with_airport_info['efficiency_score'] = final_grouped_with_airport_info['total_delay'].apply(calculate_average)

# final_grouped_with_airport_info = final_grouped_with_airport_info.drop(columns=['total_delay'])

final_grouped_with_airport_info
final_grouped_with_airport_info.columns = list(map(lambda x: str(x).lower(), final_grouped_with_airport_info.columns))

final_grouped_with_airport_info.to_csv('airport_data.csv', index=False)