# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = {

    'tra': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_visit_data.csv'),

    'as': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_store_info.csv'),

    'hs': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_reserve.csv'),

    'hr': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/hpg_reserve.csv'),

    'id': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/store_id_relation.csv'),

    'tes': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/sample_submission.csv'),

    'hol': pd.read_csv('../input/recruit-restaurant-visitor-forecasting/date_info.csv').rename(columns={'calendar_date':'visit_date'})

    }



data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])



for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']).dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime']).dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])



data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date



data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

data['tes']['visit_date'] = data['tes']['visit_date'].dt.date



unique_stores = data['tes']['air_store_id'].unique()

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#OPTIMIZED BY JEROME VALLET

tmp = data['tra'].groupby(['air_store_id','dow']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()

tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 



stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 

# NEW FEATURES FROM Georgii Vyshnia

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

lbl = preprocessing.LabelEncoder()

for i in range(10):

    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])



data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 



train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



for df in ['ar','hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])



train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)



train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']

train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2

train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2



test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']

test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2

test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2



# NEW FEATURES FROM JMBULL

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

train['var_max_lat'] = train['latitude'].max() - train['latitude']

train['var_max_long'] = train['longitude'].max() - train['longitude']

test['var_max_lat'] = test['latitude'].max() - test['latitude']

test['var_max_long'] = test['longitude'].max() - test['longitude']



# NEW FEATURES FROM Georgii Vyshnia

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 

test['lon_plus_lat'] = test['longitude'] + test['latitude']



lbl = preprocessing.LabelEncoder()

train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])

test['air_store_id2'] = lbl.transform(test['air_store_id'])



col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]

train = train.fillna(-1)

test = test.fillna(-1)
train.visit_date = pd.to_datetime(train.visit_date)

test.visit_date = pd.to_datetime(test.visit_date)
def add_weather(dataset):                                                                                                                     

    print('Adding weather...')                                                                                                                

    air_nearest = pd.read_csv(                                                                                                                

        '../input/rrv-weather-data/air_store_info_with_nearest_active_station.csv')                                                              

    unique_air_store_ids = list(dataset.air_store_id.unique())                                                                                

                                                                                                                                              

    weather_dir = '../input/rrv-weather-data/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather/'                                                                            

    weather_keep_columns = ['precipitation', 'avg_temperature']                                                                                                                                   

                                                                                                                                              

    dataset_with_weather = dataset.copy()                                                                                                     

    for column in weather_keep_columns:                                                                                                       

        dataset_with_weather[column] = np.nan                                                                                                 

                                                                                                                                              

    for air_id in unique_air_store_ids:                                                                                                       

        station = air_nearest[air_nearest.air_store_id == air_id].station_id.iloc[0]                                                          

        weather_data = pd.read_csv(weather_dir + station + '.csv', parse_dates=['calendar_date']).rename(columns={'calendar_date': 'visit_date'})                                                                                                                                           

                                                                                                                                              

        this_store = dataset.air_store_id == air_id                                                                                           

        merged = dataset[this_store].merge(weather_data, on='visit_date', how='left')                                                         

                                                                                                                                              

        for column in weather_keep_columns:                                                                                                   

            dataset_with_weather.loc[this_store, column] = merged[column]                                                                     

    return dataset_with_weather                                                                                                               

                                                                                                                                              

train = add_weather(train)                                                                                                                    

test = add_weather(test)  
train.head()