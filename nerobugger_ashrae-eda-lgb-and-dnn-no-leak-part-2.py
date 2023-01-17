import gc

import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

from tqdm import tqdm  

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



import tensorflow.keras

from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Sequential, Model, load_model

import tensorflow.keras.backend as K

from tensorflow.keras.layers import *

from tensorflow.keras import optimizers

from tensorflow.keras.losses import mean_squared_error as mse_loss

from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import PolynomialFeatures

from tensorflow.keras.regularizers import l2

from sklearn.model_selection import StratifiedKFold

def fill_weather_dataset(weather_df, mode_type):

    

    weather_df.loc[:,'timestamp'] = weather_df['timestamp'].astype(str)

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True) 



         



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    weather_df["hour"] = weather_df["datetime"].dt.hour

    weather_df["weekday"] = weather_df["datetime"].dt.weekday

#    

    #Use IterativeImputer to fill missing value 

#    df_weather_timestamp = weather_df.timestamp

#    weather_df = weather_df.drop(['timestamp','datetime'],axis=1)

#    imp = IterativeImputer(max_iter=20, random_state=0)

#    df_weather_train_np = imp.fit_transform(weather_df)

#    weather_df = pd.DataFrame(df_weather_train_np, columns=weather_df.columns)

#    weather_df.loc[:,'timestamp'] = df_weather_timestamp

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    air_temperature_filler = air_temperature_filler.fillna(method='ffill')

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    due_temperature_filler = pd.DataFrame(due_temperature_filler.fillna(method='ffill'),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    wind_direction_filler =  pd.DataFrame(wind_direction_filler.fillna(method='ffill'),columns=["wind_direction"])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    wind_speed_filler =  pd.DataFrame(wind_speed_filler.fillna(method='ffill'),columns=["wind_speed"])

    weather_df.update(wind_speed_filler,overwrite=False)

     



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    if mode_type == 'Dnn':

        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                    "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                    "2019-01-01"] 

        weather_df["is_holiday"] = (weather_df.datetime.dt.date.astype("str").isin(holidays)).astype(int)

        

        beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

        for item in beaufort:

            weather_df.loc[(weather_df['wind_speed']>=item[1]) & (weather_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]





 

    weather_df = weather_df.drop(['offset','datetime'],axis=1) 

    weather_df = weather_df.reset_index()     





    return weather_df

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def data_building_processing(df_data):

    '''===========Building data processing======================'''

    print('Processing building data...')



    lbl = LabelEncoder() 

    lbl.fit(list(df_data['primary_use'].values)) 

    df_data['primary_use'] = lbl.transform(list(df_data['primary_use'].values))

    imp = IterativeImputer(max_iter=30, random_state=0)

    df_build = imp.fit_transform(df_data)

    df_data = pd.DataFrame(df_build, columns=df_data.columns)

    df_data.loc[:,'floor_count'] = df_data['floor_count'].apply(int)

    df_data.loc[:,'year_built'] = df_data['year_built'].apply(int)  



#    df_data['year_built_1920'] = df_data['year_built'].apply(lambda x: 1 if x<1920 else 0 )

#    df_data['year_built_1920_1950'] = df_data['year_built'].apply(lambda x: 1 if 1920<=x & x<1950 else 0 )

#    df_data['year_built_1950_1970'] = df_data['year_built'].apply(lambda x: 1 if 1950<=x & x<1970 else 0 )

#    df_data['year_built_1970_2000'] = df_data['year_built'].apply(lambda x: 1 if 1970<=x & x<2000 else 0 )

#    df_data['year_built_2000'] = df_data['year_built'].apply(lambda x: 1 if x>=2000 else 0 )

    return df_data
def features_engineering(df, mode_type):

    



    classify_columns = ['building_id','meter','site_id','primary_use',

                        'hour','weekday'] 

    # Sort by timestamp

    df.sort_values("timestamp")

    df.reset_index(drop=True)

    

    # Add more features

    df['square_feet'] =  np.log1p(df['square_feet'])

    

    drop = ["timestamp","sea_level_pressure", "wind_direction", "wind_speed",]

    df = df.drop(drop, axis=1)

    gc.collect()

    

    # Encode Categorical Data

    for i in tqdm(classify_columns):

        le = LabelEncoder()

        df[i] = le.fit_transform(df[i])

        

    if mode_type == 'Dnn':

        numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']

        print('Start working with numerical characteristics...')

        for i in tqdm(numericals):

            ss_X=StandardScaler() 

            df.loc[:,i] = ss_X.fit_transform(df[i].values.reshape(-1, 1))    

    

    return df
def data_pro(df_data, df_weather_train, df_weather_test, df_building, data_type='train',mode_type='lgb'):

    ## REducing memory

    df_data = reduce_mem_usage(df_data,use_float16=True)

    df_building = reduce_mem_usage(df_building,use_float16=True)

    '''===Align local timestamps===='''

    weather = pd.concat([df_weather_train,df_weather_test],ignore_index=True)

    weather_key = ['site_id', 'timestamp']

    temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

    # calculate ranks of hourly temperatures within date/site_id chunks

    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

    # create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)

    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

    # Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.

    site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

    site_ids_offsets.index.name = 'site_id'

    

    def timestamp_align(df):

        df['offset'] = df.site_id.map(site_ids_offsets)

        df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))

        df['timestamp'] = df['timestamp_aligned']

        del df['timestamp_aligned']

        return df

 

    if data_type == 'test':

        print("Test data detected...")

        df_weather_test = timestamp_align(df_weather_test)

        df_weather_test = fill_weather_dataset(df_weather_test, mode_type)

        df_weather_test = reduce_mem_usage(df_weather_test,use_float16=True)

        #merge

        df_building = data_building_processing(df_building)

        df_data = pd.merge(df_data, df_building, on='building_id', how='left')

        df_data = df_data.merge(df_weather_test,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

    

        print("Start feature processing...")    

        df_data = features_engineering(df_data, mode_type)

        gc.collect()

        

        return df_data

    

    elif data_type == 'train':

        print("Train data detected...")

        df_weather_train = timestamp_align(df_weather_train)

        df_weather_train = fill_weather_dataset(df_weather_train, mode_type)

        df_weather_train = reduce_mem_usage(df_weather_train,use_float16=True)

        #merge

        

        df_building = data_building_processing(df_building)

        df_data = df_data.merge(df_building, left_on='building_id',right_on='building_id',how='left')

        df_data = df_data.merge(df_weather_train,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

        target_data = df_data['meter_reading']

        df_data = df_data.drop('meter_reading',axis=1)

        print("Start feature processing...")

        

        df_data = features_engineering(df_data, mode_type)

        gc.collect()



        return df_data, target_data
def cal_rmsle(Ytrue, Yfit):

    rmsle = K.sqrt(K.mean(K.square(Ytrue - Yfit), axis=0))

    return (rmsle) 



def plot_history(network_history):

    plt.figure()

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.plot(network_history.history['loss'])

    plt.plot(network_history.history['val_loss'])

    plt.legend(['Training', 'Validation']) 

    plt.figure()

    plt.xlabel('Epochs')

    plt.ylabel('rmsle')

    plt.plot(network_history.history['cal_rmsle'])

    plt.plot(network_history.history['val_cal_rmsle'])

    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.show()

    

def get_keras_data(df, num_cols, cat_cols):

    cols = num_cols + cat_cols

    X = {col: np.array(df[col]) for col in cols}

    return X



def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold, model_save_path, patience=3):

    early_stopping = EarlyStopping(patience=patience, verbose=1)

    model_checkpoint = ModelCheckpoint(model_save_path+"model_" + str(fold) + ".hdf5",

                                       save_best_only=True, verbose=1, monitor='val_loss', mode='min')



    history = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,

                            validation_data=(X_v, y_valid), verbose=1,

                            callbacks=[early_stopping, model_checkpoint])

    #draw 

    plot_history(history)

    keras_model = load_model(model_save_path+"model_" + str(fold) + ".hdf5", 

                             custom_objects={'cal_rmsle': cal_rmsle,})

    

    return keras_model
train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

train_df = train_df[train_df['building_id'] != 1099 ]

train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_df = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=["timestamp"],)

df_weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=["timestamp"],)

train_data,target_data = data_pro(train_df, weather_df, df_weather_test, building_df, data_type='train', mode_type='Dnn')

target_data = np.log1p(target_data)
def model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, 

dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001):



    #Inputs

    site_id = Input(shape=[1], name="site_id")

    building_id = Input(shape=[1], name="building_id")

    meter = Input(shape=[1], name="meter")

    primary_use = Input(shape=[1], name="primary_use")

    square_feet = Input(shape=[1], name="square_feet")

    year_built = Input(shape=[1], name="year_built")

    air_temperature = Input(shape=[1], name="air_temperature")

    cloud_coverage = Input(shape=[1], name="cloud_coverage")

    dew_temperature = Input(shape=[1], name="dew_temperature")

    hour = Input(shape=[1], name="hour")

    precip = Input(shape=[1], name="precip_depth_1_hr")

    weekday = Input(shape=[1], name="weekday")

    beaufort_scale = Input(shape=[1], name="beaufort_scale")

    day = Input(shape=[1], name="day")

    month = Input(shape=[1], name="month")

    week = Input(shape=[1], name="week")

    is_holiday = Input(shape=[1], name="is_holiday")

    floor_count = Input(shape=[1], name="floor_count")

   

    #Embeddings layers

    emb_site_id = Embedding(16, 2,name="emb_site_id")(site_id)

    emb_building_id = Embedding(1449, 6, name="emb_building_id")(building_id)

    emb_meter = Embedding(4, 2, name="emb_meter")(meter)

    emb_primary_use = Embedding(16, 2, name="emb_primary_use")(primary_use)

    emb_hour = Embedding(24, 3, name="emb_hour")(hour)

    emb_weekday = Embedding(7, 2, name="emb_weekday")(weekday)

    emb_month = Embedding(13, 2, name="emb_month")(month)



    concat_emb = concatenate([

           Flatten() (emb_site_id)

         , Flatten() (emb_building_id)

         , Flatten() (emb_meter)

         , Flatten() (emb_primary_use)

         , Flatten() (emb_hour)

         , Flatten() (emb_weekday)

         , Flatten() (emb_month)

    ])

    

    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb))

    categ = BatchNormalization()(categ)

    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))

    

    #main layer

    main_l = concatenate([

          categ

        , square_feet

        , year_built

        , air_temperature

        , cloud_coverage

        , dew_temperature

        , precip

        , beaufort_scale

        ,day

        ,week

        ,is_holiday

        ,floor_count

    ])

    

    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))

    main_l = BatchNormalization()(main_l)

    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))

    

    #output

    output = Dense(1) (main_l)



    model = Model([ site_id,

                    building_id, 

                    meter, 

                    primary_use, 

                    square_feet, 

                    year_built, 

                    air_temperature,

                    cloud_coverage,

                    dew_temperature, 

                    hour,

                    weekday, 

                    precip,

                    beaufort_scale,

                    day,

                    month,

                    week,

                    is_holiday,

                    floor_count], output)



    model.compile(optimizer = optimizers.Adam(lr=lr),

                  loss= mse_loss,

                  metrics=[cal_rmsle])

    return model


models = []

lr=0.001

batch_size = 1024

epochs = 4

folds = 2

seed = 2019

model_save_path = "../"

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']

categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday",  "meter",'day','month','week','is_holiday',]

all_unique = dict(zip([i for i in categoricals],[train_data[i].unique() for i in categoricals]))



for fold_n, (train_index, valid_index) in enumerate(kf.split(train_data, train_data['building_id'])):

    print('Fold:', fold_n) 

    X_train, X_valid = train_data.iloc[train_index], train_data.iloc[valid_index]

    y_train, y_valid = target_data.iloc[train_index], target_data.iloc[valid_index]

    X_t = get_keras_data(X_train , numericals, categoricals)

    X_v = get_keras_data(X_valid, numericals, categoricals)

    keras_model = model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, 

                        dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=lr)

    mod = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold_n, model_save_path, patience=3)

    models.append(mod)

    print('*'* 50)

    del(X_train, X_valid, y_train, y_valid, X_t, X_v)

    gc.collect()

    

del(train_df, weather_df, df_weather_test, building_df)
#Load test data

#df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

#row_ids = df_test["row_id"]

#df_test.drop("row_id", axis=1, inplace=True)

#building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

#weather_df = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=["timestamp"],)

#df_weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=["timestamp"],)

#df_test = data_pro(df_test, weather_df, df_weather_test, building_df, data_type='test', mode_type='Dnn')

#df_test.loc[df_test['beaufort_scale'] == 9,'beaufort_scale'] = 8





#i=0

#target_tests = np.zeros((df_test.shape[0]),dtype=np.float32)

#step_size = 5000

#for j in tqdm(range(int(np.ceil(df_test.shape[0]/step_size)))):

#    for_prediction = get_keras_data(df_test.iloc[i:i+step_size],numericals, categoricals)

#    target_tests[i:min(i+step_size,df_test.shape[0])] = \

#        np.expm1(sum([model.predict(for_prediction, batch_size=1024)[:,0] for model in models])/len(models))

#    i+=step_size



#results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(target_tests, 0, a_max=None)})

#results_df['meter_reading'] = results_df['meter_reading'].round(4)

#del row_ids,target_tests

#gc.collect()

#results_df.to_csv("submission.csv", index=False)