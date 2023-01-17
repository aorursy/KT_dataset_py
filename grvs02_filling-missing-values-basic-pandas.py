import pandas as pd

df = pd.read_csv(r'../input/weather_place.data.csv',parse_dates=['date'])

df.head()
df.drop(columns = 'Unnamed: 0' , inplace = True)

df.head()
df.isnull().values.any()
new_df = df.fillna(0)

new_df
new_df = df.fillna({

        'temperature': 0,

        'windspeed': 0,

        'event': 'no event'

    })

new_df
new_df = df.fillna(method="ffill")

new_df