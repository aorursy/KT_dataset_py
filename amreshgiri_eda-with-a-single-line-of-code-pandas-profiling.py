import pandas_profiling as pp

import pandas as pd 
df_chennai_rainfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv', parse_dates=['Date'], index_col='Date')
df_report = pp.ProfileReport(df_chennai_rainfall)

df_report
df_chennai_reservoirs = pd.read_csv('../input/chennai_reservoir_levels.csv', parse_dates=['Date'], index_col='Date')
df_report = pp.ProfileReport(df_chennai_reservoirs)

df_report