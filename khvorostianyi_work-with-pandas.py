K=33; N = 11 
MONTH_N = 3 % 12 + 1 ;YEAR_YYYY = 1998 + 23 - 14
YEAR_ZZZZ = 2015 - (23 - 14); MONTH_M = 12-11+1
import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('../input/weather_madrid_LEMD_1997_2015.csv', index_col='CET', parse_dates=True)
df.head(K+N)
df.columns.values
df[' Events'].unique()
df_3th_t=df.loc[f'{YEAR_YYYY}-{MONTH_N}']
df_3th_complete = df_3th_t[df_3th_t[' Events'].str.contains('Rain') == True]
print(f'qty of rainy days:{len(df_3th_complete)}')
df_3th_complete[[' Events']]
df_3th_t[[' Mean Humidity',' Min Humidity']]
df_4th_t = df.loc[f'{YEAR_ZZZZ}-{MONTH_M}'].copy()
df_4th_t.head()
df_4th_t['delta_temp'] = df_4th_t['Max TemperatureC']-df_4th_t['Min TemperatureC']
df_4th_t[['delta_temp','Max TemperatureC','Min TemperatureC']].head()

df_4th_t[['delta_temp','Precipitationmm']].plot(title='Delta Temperature(Celcius) and Precipitation(mm)',
                                                figsize=(15,7))