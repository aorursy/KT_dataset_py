import numpy as np 

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import norm

import time

from sklearn.impute import SimpleImputer

import pandas as pd

import numpy as np

from scipy.stats.stats import pearsonr  

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from sklearn.model_selection import GridSearchCV
station= pd.read_csv('../input/station.csv')

status= pd.read_csv('../input/status.csv')

trip= pd.read_csv('../input/trip.csv')

weather= pd.read_csv('../input/weather.csv')
trip.duration = trip.duration/60

trip = trip[trip.duration <= 360]


trip.start_date = pd.to_datetime(trip.start_date, format='%m/%d/%Y %H:%M')
trip['date'] = pd.to_datetime(trip.start_date)

trip.date=trip.date.dt.strftime('%m/%d/%Y')


trip['station_date'] = trip['date'].map(str) + " "+trip['start_station_name']

cols = list(trip)

cols.insert(0, cols.pop(cols.index('station_date')))

trip = trip.ix[:, cols]
#Each entry in the date feature is a trip. 

#By finding the total number of times a date is listed, we know how many trips were taken on that date.

station_dates = {}

for d in trip.station_date:

    if d not in station_dates:

        station_dates[d] = 1

    else:

        station_dates[d] += 1
#Create the data frame that will be used for training, with the dictionary we just created.

df2 = pd.DataFrame.from_dict(station_dates, orient = "index")

df2['station_date'] = df2.index

df2['trips'] = df2.ix[:,0]

train = df2.ix[:,1:3]

train.reset_index(drop = True, inplace = True)
trip = pd.merge(trip, station, left_on='start_station_id', right_on='id')

trip['in_city'] = np.where(trip['lat'] >37.5630, 1, 0 )

trip['zip_to_use'] = np.where(trip['lat'] >37.5630, 95113, 94107 )
merge1 = pd.merge(train, trip, left_on='station_date', right_on='station_date')
merge2 = merge1.drop_duplicates(subset='station_date', keep='first')
weather.drop(weather[weather.zip_code == 94063].index, inplace=True)

weather.drop(weather[weather.zip_code == 94301].index, inplace=True)

weather.drop(weather[weather.zip_code == 94041].index, inplace=True)

weather = weather.drop(['max_gust_speed_mph'],1)

events = pd.get_dummies(weather.events)

weather = weather.merge(events, left_index = True, right_index = True)

weather = weather.drop(['events'],1)
#Change this feature from a string to numeric.

#Use errors = 'coerce' because some values currently equal 'T' and we want them to become NAs.

weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors = 'coerce')
weather.loc[weather.precipitation_inches.isnull(), 

            'precipitation_inches'] = weather[weather.precipitation_inches.notnull()].precipitation_inches.median()
merge2['merge_key'] = merge2['zip_to_use'].map(str) + " "+merge2['date'].map(str)

weather['merge_key'] = weather['zip_code'].map(str) + " "+weather['date'].map(str)



merge2['merge_key'] = weather['merge_key'].str.strip()

merge3 = pd.merge(merge2, weather,   left_on=['merge_key'], right_on = ['merge_key' ], how='left')
merge3.drop(['id_x','duration', 'start_date', 'start_station_name', 'start_station_id', 'end_date','end_station_name', 

             'end_station_id'],1, inplace= True)

merge3.drop(['id_y','name', 'lat', 'long', 'city','installation_date', 'zip_to_use', 'merge_key'],1, inplace= True)

merge3.drop(['bike_id','zip_code_y'],1, inplace= True)

# #Find all of the holidays during our time span

calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=merge3.date_x.min(), end=merge3.date_x.max())
#Find all of the business days in our time span

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=merge3.date_x.min(), end=merge3.date_x.max(), freq=us_bd)
business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date

holidays = pd.to_datetime(holidays, format='%Y/%m/%d').date
#A 'business_day' or 'holiday' is a date within either of the respected lists.



merge3['business_day'] = merge3.date_x.isin(business_days)

merge3['holiday'] = merge3.date_x.isin(holidays)
# #Convert True to 1 and False to 0

merge3.business_day = merge3.business_day.map(lambda x: 1 if x == True else 0)

merge3.holiday = merge3.holiday.map(lambda x: 1 if x == True else 0)

merge3.subscription_type = merge3.subscription_type.map(lambda x: 1 if x == 'Subscriber'else 0)

merge3.drop(['station_date', 'date_x', 'date_y'], 1, inplace=True)

merge3.drop(['zip_code_x'], 1, inplace=True)





labels = merge3.trips

merge4 = merge3.drop(['trips'], 1)

my_imputer = SimpleImputer()

merge4 = my_imputer.fit_transform(merge3)
train_set, test_set, y_train, y_test = train_test_split(merge4, labels, test_size=0.2)
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],

'max_depth': [4, 6],

'min_samples_leaf': [3, 5, 9, 17],



}

est = GradientBoostingRegressor(n_estimators=500, random_state=33)


gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(train_set, y_train)
gs_cv.score(test_set, y_test)
gs_cv.best_params_
best_gs = GradientBoostingRegressor(learning_rate=.1, max_depth=6, min_samples_leaf=3, n_estimators=1000)
best_gs.fit(train_set, y_train)
best_gs.score(test_set, y_test)