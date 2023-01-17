import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

%matplotlib inline
trip = pd.read_csv("../input/trip.csv")
weather = pd.read_csv("../input/weather.csv")
station = pd.read_csv("../input/station.csv")
def getDate(df):
    for d in df:
        yield d.date()

def getHour(df):
    for d in df:
        yield d.hour

def getWeekday(df):
    for d in df:
        yield d.weekday()

def getMonth(df):
    for d in df:
        yield d.month
station.index = station.id
station = station.drop('id', axis=1)
station.head()
station.plot.barh(x= 'name', y='dock_count',figsize=(10,10))
trip.index = trip.id
trip = trip.drop('id', axis=1)

dates = pd.to_datetime(trip.start_date, format='%m/%d/%Y %H:%M')
trip['datetime'] = dates
trip['date'] = list(getDate(dates))
trip['month'] = list(getMonth(dates))
trip['hour'] = list(getHour(dates))
trip['weekday'] = list(getWeekday(dates))
trip['zip_code'] = trip.zip_code.astype(str)

trip = trip[trip.duration <= 60*60]

trip.head()
weather.index = pd.to_datetime(weather.date)
weather['date'] = list(getDate(weather.index))
weather['weekday'] = list(getWeekday(weather.index))

weather.precipitation_inches = weather.precipitation_inches.replace('T', 0.01)
weather.precipitation_inches = weather.precipitation_inches.astype(float)

weather.zip_code = weather.zip_code.astype(str)

tmp_weather = pd.get_dummies(weather.events, drop_first=True)
tmp_weather['Rain'] = tmp_weather.Rain + tmp_weather.rain
weather = pd.concat([tmp_weather, weather],axis=1)
weather = weather.drop('rain',axis=1)

weather.head()
df = weather.merge(trip.drop('weekday', axis=1), on=['date', 'zip_code'], how='left')
df.head()
df = df[df.duration.notnull()]
df_dur = df.drop(['date', 'events', 'zip_code', 'start_date', 'start_station_name', 'start_station_id', 'weekday',
                  'end_date', 'end_station_name', 'end_station_id', 'bike_id', 'subscription_type', 'datetime', 'month', 'hour'], axis=1)

stdc_dur = StandardScaler()
tmp_df_dur = df_dur.drop(['Fog-Rain', 'Rain', 'Rain-Thunderstorm'], axis=1)
tmp_df_dur = tmp_df_dur.fillna(method='ffill')
df_dur_std = pd.DataFrame(stdc_dur.fit_transform(tmp_df_dur), columns=tmp_df_dur.columns, index = df_dur.index)

for c in tmp_df_dur.columns:
    df_dur[c] = df_dur_std[c]

# get dummy variables
df_weekday = pd.get_dummies(df.weekday, drop_first=True, prefix='weekday_')
df_month = pd.get_dummies(df.month, drop_first=True, prefix='month_')
df_hour = pd.get_dummies(df.hour, drop_first=True, prefix='hour_')
df_subscription_type = pd.get_dummies(df.subscription_type, drop_first=True)

df_dur = pd.concat([df_dur, df_weekday, df_month, df_hour, df_subscription_type], axis=1)
df_dur.index = df.date
df_dur.head()
df_dur_train = df_dur.iloc[:91153,:]
df_dur_test = df_dur.iloc[91153:,:]
df.weekday.hist(bins=7)
df.month.hist(bins=12)
df.hour.hist(bins=24)
dur = df.duration / 60
dur.hist(bins=60)
df.plot.scatter('mean_temperature_f', 'duration')
df.events.value_counts().plot.bar()
tmp_df = df[['hour', 'duration']].dropna()
tmp_df['hour'] = tmp_df['hour'].astype(int)
sns.boxplot(x='hour', y='duration', data=tmp_df)
freq_trip = trip.groupby(['date','zip_code', 'start_station_id']).size()
freq_trip.rename('cnt', inplace=True)
freq_trip = freq_trip.to_frame().reset_index()
df2 = weather.merge(freq_trip, on=['date', 'zip_code'], how='left')

df2.start_station_id = df2.start_station_id.fillna(0)
df2.start_station_id = df2.start_station_id.astype(int)

df2.cnt = df2.cnt.fillna(0)

df2['month'] = list(getMonth(df2['date']))

df2.head()
df2.shape
df_cnt = df2.drop(['date', 'events', 'zip_code', 'start_station_id', 'weekday', 'month'], axis=1)

stdc_cnt = StandardScaler()
tmp_df_cnt = df_cnt.drop(['Fog-Rain', 'Rain', 'Rain-Thunderstorm'], axis=1)
tmp_df_cnt = tmp_df_cnt.fillna(method='ffill')
df_cnt_std = pd.DataFrame(stdc_cnt.fit_transform(tmp_df_cnt), columns=tmp_df_cnt.columns, index = df_cnt.index)

for c in tmp_df_cnt.columns:
    df_cnt[c] = df_cnt_std[c]

# get dummy variables
df_weekday = pd.get_dummies(df2.weekday, drop_first=True, prefix='weekday_')
df_month = pd.get_dummies(df2.month, drop_first=True, prefix='month_')
df_start_station_id = pd.get_dummies(df2.start_station_id, drop_first=True, prefix='start_station_id_')

df_cnt = pd.concat([df_cnt, df_weekday, df_month, df_start_station_id], axis=1)
df_cnt.index = df2.date
df_cnt.head()
X, Y = df_cnt.drop('cnt', axis=1).values, df_cnt['cnt'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

df_cnt_train = pd.DataFrame(X_train, columns= df_cnt.columns.drop('cnt'))
df_cnt_train['cnt'] = Y_train

df_cnt_test = pd.DataFrame(X_test, columns= df_cnt.columns.drop('cnt'))
df_cnt_test['cnt'] = Y_test
df_cnt_train = df_cnt.iloc[:32531,:]
df_cnt_test = df_cnt.iloc[32531:,:]
df2.cnt.hist(bins=30)
tmp_df2 = df2.drop(['date', 'events', 'zip_code'], axis=1).fillna(method='ffill')
tmp_df2.head()
sns.heatmap(df_cnt.corr())
status.head()
