
#Import all the stuff we need



import numpy as np 

import pandas as pd 

import xgboost as xgb



from bs4 import BeautifulSoup

from urllib.request import urlopen

from datetime import datetime, date, timedelta
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
weather = pd.read_csv('../input/nyc-taxi-wunderground-weather/weatherdata.csv')

# Full date isn't very useful, so let's just number the days

def get_days_delta(datestr):

    datetimeobj = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')

    return (date(2016, 6, 30) - datetimeobj.date()).days



# also grab the hour

def get_hour(datestr):

    datetimeobj = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')

    return datetimeobj.hour



# and the minute

def get_minute(datestr):

    datetimeobj = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')

    return datetimeobj.minute



# the distance between the two - don't bother with curavture of the earth effects

def get_dist(plong, plat, dlong, dlat):

    return np.sqrt((plong - dlong)**2 + 

                    (plat - dlat)**2)



# the "manahatten distance", which is lengths added, not in quadriture

# note: probably shouldn't just be lats and longs added, unless streets really

# do run exactly N/S and E/W

def get_mdist(plong, plat, dlong, dlat):

    return np.abs(plong - dlong) + np.abs(plat - dlat)



# the direction of each trip

def get_dir(plong, plat, dlong, dlat):

    return np.arctan2((plat - dlat), (plong - dlong))



# whether it's a holiday or not

def is_hol(datestr):

    datetimeobj = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')

    holiday_list = [date(2016, 1, 1), date(2016, 1, 18), date(2016, 2, 15), date(2016, 5, 30)]



    return datetimeobj.date() in holiday_list



# Here's where they all get added. Pretty self-explanatory, note I also do the log of each distance

# as well as the day of the week

def add_columns(df, weather):

    df['day'] = np.vectorize(get_days_delta)(df['pickup_datetime'])

    df['dow'] = np.vectorize(lambda x: x%7)(df['day'])

    df['hour'] = np.vectorize(get_hour)(df['pickup_datetime'])

    df['min'] = np.vectorize(get_minute)(df['pickup_datetime'])

    df['is_hol'] = np.vectorize(is_hol)(df['pickup_datetime'])

    df['storeflag'] = np.vectorize(lambda x: 0 if x=='N' else 1)(df['store_and_fwd_flag'])

    df['dist'] = np.vectorize(get_dist)(df['pickup_longitude'], 

                                               df['pickup_latitude'], 

                                               df['dropoff_longitude'], 

                                               df['dropoff_latitude'])

    df['mdist'] = np.vectorize(get_mdist)(df['pickup_longitude'], 

                                                 df['pickup_latitude'], 

                                                 df['dropoff_longitude'], 

                                                 df['dropoff_latitude'])

    df['logdist'] = np.log(df['dist'] + 1)

    df['logmdist'] = np.log(df['mdist'] + 1)

    df['dir'] = np.vectorize(get_dir)(df['pickup_longitude'], 

                                               df['pickup_latitude'], 

                                               df['dropoff_longitude'], 

                                               df['dropoff_latitude'])

    

    # Now here is where the weather data is added

    # Data frame must be sorted by date beforehand

    # It starts with the first trip, scans through weather data until it passes

    # the time of the trip start, then uses the next point. Starts with that point for

    # next trip

    start_index = 0

    trip_precips = []

    trip_temps = []



    for ele in df['pickup_datetime']:

        ts = datetime.strptime(ele, '%Y-%m-%d %H:%M:%S')

        while True:

            if ts > datetime.strptime(weather['timestamp'][start_index], '%Y-%m-%d %H:%M:%S'):

                start_index += 1

            else:

                trip_precips.append(weather['precip'][start_index])

                trip_temps.append(weather['temp'][start_index])

                break

    

    # For speed, the data is initially put into lists, then added to dataframe later

    df['temp'] = trip_temps

    df['precip'] = trip_precips

    df['isprecip'] = df['precip'] > 0
add_columns(train, weather)
add_columns(test, weather)

# very important: since the evaluation metric is the RMSLE, predict the log of the trip duration
train['logtime'] = np.log(train['trip_duration'] + 1)
cols = ['vendor_id', 
        'passenger_count', 
        'pickup_longitude', 
        'pickup_latitude', 
        'dropoff_longitude', 
        'dropoff_latitude', 
        'day', 
        'dow', 
        'hour', 
        'min', 
        'dist', 
        'dir', 
        'temp', 
        'precip', 
        #'logdist',
        'storeflag', 
        'is_hol',
        'mdist',
        #'logmdist'
       ]
ts = train.sample(frac=1)

cvtrain = ts[:1200000][cols]
cvtraintimes = ts[:1200000]['logtime']
cvverify = ts[1200000:1300000][cols]
cvverifytimes = ts[1200000:1300000]['logtime']
cvverify2 = ts[1300000:][cols]
cvverifytimes2 = ts[1300000:]['logtime']

xgbtrain = xgb.DMatrix(cvtrain, label=cvtraintimes)
xgbverify = xgb.DMatrix(cvverify, label=cvverifytimes)
xgbverify2 = xgb.DMatrix(cvverify2, label=cvverifytimes2)
xgbtest = xgb.DMatrix(test[cols])
xgbparams = {'max_depth':10, 
               'n_estimators':3000,
               'learning_rate':0.035,
               'subsample':0.8, 
               'tree_method': 'exact',
               'alpha':5,
               'lambda':10
               }

num_round = 2000

mdl = xgb.train(xgbparams, xgbtrain, num_round, [(xgbtrain, 'train'), (xgbverify, 'verify')], early_stopping_rounds=10)
def rmsle(predictor, X, y):
    return np.sqrt(np.mean((predictor.predict(X) - y)**2))

print(rmsle(mdl, xgb.DMatrix(cvverify2), cvverifytimes2))
preds = mdl.predict(xgbtest)
xgb_test_preds_frame = pd.DataFrame({'trip_duration': np.exp(preds) - 1}, index=test['id'])
xgb_test_preds_frame.to_csv('xgb_test_preds_frame.csv')