import pandas as pd
df= pd.read_csv('../input/weather-forecast-report/nyc_weather.csv')
df
df['EST'][df['Events']=='Snow']
df[(df['WindSpeedMPH']>=5)&(df['Humidity']<=60)][['Temperature','EST']]
df[(df['PrecipitationIn']!='0')][['WindSpeedMPH','Humidity','DewPoint']].plot(kind='bar',figsize=[15,4])
df['WindSpeedMPH']
df['WindSpeedMPH'].mean()
df['WindSpeedMPH'].fillna(0)
