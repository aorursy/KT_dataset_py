import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/new-york-city-taxi-trip-hourly-weather-data/Weather.csv')
df2 = df.set_index('pickup_datetime')[['tempi']]

plt.figure(figsize=(16,4))
df2['tempi'].plot()
# Switch month and day when day is less than 12
df['date'] = pd.to_datetime(df['pickup_datetime'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute

def compute_right_day(row):
    if row['day'] <= 12:
        return row['month']
    else:
        return row['day']

def compute_right_month(row):
    if row['day'] <= 12:
        return row['day']
    else:
        return row['month']

def compute_right_date(row):
    return '{:d}-{:d}-{:d} {:d}:{:d}'.format(row['year'], row['new_month'], row['new_day'],
                                             row['hour'], row['minute'])

df['new_day'] = df.apply(compute_right_day, axis=1)
df['new_month'] = df.apply(compute_right_month, axis=1)
df['new_date_str'] = df.apply(compute_right_date, axis=1)
df['date'] = pd.to_datetime(df['new_date_str'])
df = df[['date', 'tempi']]
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
plt.figure(figsize=(16,4))
df['tempi'].plot()
