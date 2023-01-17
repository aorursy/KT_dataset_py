import numpy as np 
import pandas as pd 
from random import randint
from subprocess import check_output
from datetime import datetime
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import linear_model
weather_df = pd.read_csv('../input/weather.csv')
trip_df = pd.read_csv('../input/trip.csv')
weather_df['date'] = weather_df['date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
trip_df['start_date'] = trip_df['start_date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M"))
def sliceWeekdayAndWeekend(df, on='date'):
    weekday_mask = df[on].weekday() < 5
    weekend_mask = df[on].weekday() >= 5
    return df.loc[weekday_mask], df.loc[weekend_mask]
def sliceGoodAndBadWeatherDay(df):
    bad_weather_mask = df['events'] < 0
    good_weather_mask = df['events'] >= 0

    return df.loc[bad_weather_mask], df.loc[good_weather_mask]
def convertEventToInt(val):
    if val is np.nan:
        return 0
    elif ('Rain' in val) or ('Thunder' in val):
        return -1
    else:
        return 2
trip = trip_df.copy()
weather = weather_df.copy()
weather['events'] = weather['events'].apply(convertEventToInt)
weather['precipitation_inches'] = weather['precipitation_inches'].apply(lambda x: 0.01 if x == 'T' else x)
weather['precipitation_inches'] = weather['precipitation_inches'].astype('float64')

trip['date'] = trip['start_date'].apply(lambda x: x.date())
weather['date'] = weather['date'].apply(lambda x: x.date())

weather['zip_code'] = weather['zip_code'].astype('str')
weather.fillna(weather.mean(), inplace=True)
count_per_day = trip.groupby(['date', 'zip_code']).size()
count_per_day.rename('count', inplace=True)
count_per_day = count_per_day.to_frame().reset_index()
whole_dataset = weather.merge(count_per_day, on=['date', 'zip_code'])
whole_dataset['isWeekend'] = whole_dataset['date'].apply(lambda x: False if x.weekday() < 5 else True)
weekday_df = whole_dataset[whole_dataset['isWeekend'] == False]
bad_weather, good_weather = sliceGoodAndBadWeatherDay(weekday_df)
bad_weather.drop(['date','zip_code', 'isWeekend', 'events'], axis=1, inplace=True)
good_weather.drop(['date','zip_code', 'isWeekend', 'events'], axis=1, inplace=True)
def sliceXandY(df):
    x = df.ix[:, :'wind_dir_degrees']
    y = df.ix[:, 'count']
    return x, y
def sampleDataset(df, in_frac=0.12, in_random_state=22):
    return df.sample(frac=in_frac, random_state=in_random_state)
good_weather_sample = sampleDataset(good_weather, in_frac=0.12, in_random_state=randint(0,32767))
learning_dataset = pd.concat([good_weather_sample, bad_weather])
x, y = sliceXandY(learning_dataset)
Xs_train, Xs_test, y_train, y_test = train_test_split(scale(x), y, test_size=0.2, random_state=randint(0,32767))
lasso_model = linear_model.Lasso()
lasso_model.fit(Xs_train, y_train)
lasso_model.coef_
lasso_model.score(Xs_train, y_train)