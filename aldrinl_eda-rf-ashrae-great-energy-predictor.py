import numpy as np

import pandas as pd

import pandas_profiling



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



from sklearn.model_selection import train_test_split as train_valid_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



import eli5

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
energy = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

energy.timestamp = pd.to_datetime(energy.timestamp)

energy.meter = energy.meter.astype('category')

energy.info()
g = energy.building_id.value_counts()

plt.hist(g.values,bins=100)

plt.xlabel('Total rows of a building id')

plt.ylabel('Number building ids')

plt.show()
g = energy.meter.value_counts()

plt.bar(g.index,g.values)

plt.xlabel('Meter Type')

plt.ylabel('Count')

plt.show()
g = energy[['meter','meter_reading']]

g['meter_reading'] = np.log1p(g['meter_reading'])

sns.boxplot(x='meter',y='meter_reading',data=g)

plt.plot();
weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather.timestamp = pd.to_datetime(weather.timestamp)

weather.info()
g = weather.drop(['site_id','timestamp'],axis=1).corr()

plt.figure(figsize=(12,10))

sns.heatmap(g,annot=True,center=0,cmap='Blues');
building_info = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

building_info.info()
g = building_info.year_built.value_counts()

plt.bar(g.index,g.values)

plt.xlabel('Year Built')

plt.ylabel('Count')

plt.show()
g = building_info.primary_use.value_counts()

plt.barh(g.index,g.values)

plt.xlabel('Building Primary Use')

plt.ylabel('Count')

plt.show()
train = pd.merge(energy,building_info,on='building_id',how='left')

train = pd.merge(train,weather,on=['site_id','timestamp'],how='left')

del energy,weather

train.tail()
def dt_parts(df,dt_col):

    if(df[dt_col].dtype=='O'):

        df[dt_col] = pd.to_datetime(df[dt_col])

    df['year'] = df[dt_col].dt.year.astype(np.int16)

    df['month'] = df[dt_col].dt.month.astype(np.int8)

    df['day'] = df[dt_col].dt.day.astype(np.int8)

    df['hour'] = df[dt_col].dt.hour.astype(np.int8)

    df['minute'] = df[dt_col].dt.minute.astype(np.int8)

    df['second'] = df[dt_col].dt.second.astype(np.int8)

    df.drop(dt_col,axis=1,inplace=True)

    return df



#optimizing the column types to consume less space

def df_type_optimize(df):

    df['building_id'] = df['building_id'].astype(np.uint16)

    df['meter'] = df['meter'].astype(np.uint8)

    df['site_id'] = df['site_id'].astype(np.uint8)

    df['square_feet'] = df['square_feet'].astype(np.uint32)

    

    df['year_built'] = df['year_built'].astype(np.uint16)

    df['floor_count'] = df['floor_count'].astype(np.uint8)

    

    df['air_temperature'] = df['air_temperature'].astype(np.int16)

    df['cloud_coverage'] = df['cloud_coverage'].astype(np.int16)

    df['dew_temperature'] = df['dew_temperature'].astype(np.int16)

    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].astype(np.int16)

    df['sea_level_pressure'] = df['sea_level_pressure'].astype(np.int16)

    df['wind_direction'] = df['wind_direction'].astype(np.int16)

    df['wind_speed'] = df['wind_speed'].astype(np.int16)

    

    return df
train['primary_use'] = train['primary_use'].astype('category').cat.codes

train = dt_parts(train,'timestamp')

train.fillna(0,inplace=True)

train=df_type_optimize(train)

train.head()
target_col = 'meter_reading'

y = train[target_col]

Xs = train.drop(target_col,axis=1)



X_train, X_valid, y_train, y_valid = train_valid_split(Xs, y, test_size=0.2, random_state=0)

del train

X_train.shape,X_valid.shape
#code reference above

from sklearn.ensemble import forest

def set_rf_samples(n):

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))

set_rf_samples(130000)
%%time

model = RandomForestRegressor(n_estimators=60,

                              random_state=0,n_jobs=-1)

model.fit(X_train,y_train)
def RMSE(actual,preds):

    return np.sqrt(mean_squared_error(actual,preds))



def get_evaluations(model):

    preds = model.predict(X_train)

    plt.hist(np.log1p(preds),bins=100)

    plt.show();

    print('train_rmse: ',RMSE(y_train,preds))

                    

    preds = model.predict(X_valid)

    plt.hist(np.log1p(preds),bins=100)

    plt.show()

    print('valid_rmse: ',RMSE(y_valid,preds))

    

get_evaluations(model)
eli5.show_weights(model,feature_names=list(X_train.columns))
test_row = X_valid.loc[15256244,:]

test_row
eli5.show_prediction(model,test_row,feature_names=list(X_train.columns))
energy_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

test = pd.merge(energy_test,building_info,on='building_id',how='left')

test = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

del energy_test,weather_test

test.tail()
test['primary_use'] = test['primary_use'].astype('category').cat.codes

test = dt_parts(test,'timestamp')

test.fillna(0,inplace=True)

test=df_type_optimize(test)

ids = test['row_id']

test.drop('row_id',axis=1,inplace=True)

test.head()
%%time

preds = model.predict(test)



sub_df = pd.DataFrame()

sub_df['row_id'] = ids

sub_df['meter_reading'] = preds

sub_df.to_csv('the-sub-mission.csv',index=False)

sub_df.head()
plt.hist(np.log1p(sub_df['meter_reading']),bins=100)

plt.show()