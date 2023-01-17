import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
#import only a few rows

train = pd.read_csv('../input/train.csv', nrows=1000)
train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
df = pd.DataFrame(data=train, columns=['click_time'])
df.click_time = pd.to_datetime(df.click_time)
df['new_formatted_date'] = df.click_time.dt.strftime('%d/%m/%y %H:%M')
df.new_formatted_date.head(3)
# pandas.Series.dt

df['month'] = df.click_time.dt.month

df['day'] = df.click_time.dt.day

df['year'] = df.click_time.dt.year

df['hour'] = df.click_time.dt.hour

df['minute'] = df.click_time.dt.minute

df.head(3)
print('Unique values of month:', df.month.unique())

print('Unique values of day:', df.day.unique())

print('Unique values of year:', df.year.unique())

print('Unique values of hour:', df.hour.unique())

print('Unique values of minute:', df.minute.unique())
# Day

df['day_sin'] = np.sin(df.day*(2.*np.pi/30))

df['day_cos'] = np.cos(df.day*(2.*np.pi/30))
# Hour

df['hour_sin'] = np.sin(df.day*(2.*np.pi/24))

df['hour_cos'] = np.cos(df.day*(2.*np.pi/24))
# Minute

df['minute_sin'] = np.sin(df.day*(2.*np.pi/60))

df['minute_cos'] = np.cos(df.day*(2.*np.pi/60))
# Concatenate

concatenated = pd.concat([train, df], axis=1)
# Define X

X = concatenated[['ip', 'app', 'device', 'os', 'channel', 'day_sin','day_cos', 'hour_sin', 'hour_cos',

                  'minute_sin', 'minute_cos']]
# Now we have timestamp with sin and cosine:

X.head(3)