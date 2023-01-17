import pandas as pd
df = pd.read_csv('../input/nyc_weather.csv', parse_dates=['EST'])
df.info()
df.head()
new_df = df.fillna(method='bfill', limit=1)
new_df = df.interpolate()
new_df
new_df = df.dropna()
new_df
df['Humidity'].max()


