from tqdm import tqdm_notebook



import pandas as pd

pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',150)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from time import time

import datetime

import gc



PATH = '/kaggle/input/ashrae-energy-prediction/'
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

metadata = pd.read_csv(PATH+"building_metadata.csv",dtype=metadata_dtype)

metadata.info(memory_usage='deep')
weather_dtype = {"site_id":"uint8"}

weather_train = pd.read_csv(PATH+"weather_train.csv",parse_dates=['timestamp'],dtype=weather_dtype)

weather_test = pd.read_csv(PATH+"weather_test.csv",parse_dates=['timestamp'],dtype=weather_dtype)
train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}

train = pd.read_csv(PATH+"train.csv",parse_dates=['timestamp'],dtype=train_dtype)

test_dtype = {'meter':"uint8",'building_id':'uint16'}

test_cols_to_read = ['building_id','meter','timestamp']

test = pd.read_csv(PATH+"test.csv",parse_dates=['timestamp'],usecols=test_cols_to_read,dtype=test_dtype)
Submission = pd.DataFrame(test.index,columns=['row_id'])
train.isnull().sum()
test.isnull().sum()
metadata.isnull().sum()
weather_train.isnull().sum()
weather_test.isnull().sum()
# dim target

target = 'meter_reading'
SEED = 17



# remove unnecessary columns in the list

def set_to_list(cols, excepted):

    return list(set(cols) - set(excepted))



# get string variables

def get_update_string_variables(df, exclude_features):

        string_variables = set_to_list(list(df.select_dtypes('object').columns), exclude_features)

        return string_variables



# get null variables

def get_update_null_variables(df, exclude_features, target):

    with_null_fields = set_to_list(

        set_to_list(

            df.columns[df.isnull().any()]

            , [target]

        )

        , exclude_features

    )

    return with_null_fields



# dummies and update var's

def set_dummies(df, variables, cat_features, features):

    if len(variables)>0:

        print('Dummies preprocess for:', ', '.join(variables))

        # save change

        dummies_columns = pd.get_dummies(df[variables]).columns



        features = list(

            features + list(

                dummies_columns

            )

        )



        cat_features = list(

            cat_features + list(

                dummies_columns

            )

        )



        df = pd.get_dummies(df, columns = variables)

        features = set_to_list(features, variables)

        cat_features = set_to_list(cat_features, variables)

    return df, features, cat_features





# Model-driven Imputation w RF

# exclude object and datetime

def set_predict_null_values(

    df,

    features,

    exclude_features,

    features_for_predict,

    # ensemble = []

    type_predict, # Regressor, Classifier

    n_estimators = 30,

    n_jobs = 20,

):



    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



    if len(features) == 0:

        features = list(df.columns)

    if len(features_for_predict) == 0:

        exclude_features = []

        with_null_fields = get_update_null_variables(df, exclude_features)

    else:

        with_null_fields = features_for_predict



    from tqdm import tqdm_notebook

    if type_predict == 'Regressor':

        model = RandomForestRegressor(

            n_estimators=n_estimators, n_jobs = n_jobs, random_state = SEED

        )

    elif type_predict == 'Classifier':

        model = RandomForestClassifier(

            n_estimators=n_estimators, n_jobs = n_jobs, random_state = SEED

        )



    df_null_stat = pd.DataFrame(df[features_for_predict].isnull().sum(), columns = ['CountNull'])

    df_null_stat.sort_values('CountNull', inplace = True)

    predict_null_fields = list(df_null_stat[df_null_stat['CountNull'] > 0].index)



    print('Apply encoder for this columns, they have object or datetime variables',', '.join(df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns))

    print(list(df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns))



    for c in df[features_for_predict].select_dtypes(include=['object', 'datetime']).columns:

        if c in predict_null_fields:

            predict_null_fields.remove(c)

            print('Remove obj/datetime var:', c)



    print('\nFeatures changes:',', '.join(predict_null_fields))

    features_wo_null = set_to_list(

        df[set_to_list(features, with_null_fields)].columns

        , df.select_dtypes(include=['object', 'datetime']).columns

    )



    for c in tqdm_notebook(predict_null_fields):

        x_train = df[df[c].isnull()==False][features_wo_null]

        x_test = df[df[c].isnull()==True][features_wo_null]

        y_train = df[df[c].isnull()==False][c]



        print(

            'Predict:',c,'\nCount Null string:', x_test.shape[0],

            '\nFeatures wo null in train:', ', '.join(features_wo_null),

        )



        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        x_test[c] = y_pred



        df_without_null_c = pd.DataFrame(

            pd.concat([df[df[c].isnull()==False][c], x_test[c]])

            , columns = [c]

        )



        df[c] = df_without_null_c.sort_index()[c]



        features_wo_null.append(c)

        

    return df
df = metadata.copy()

exclude_features = []

cat_features = []

features = list(df.columns)

null_variables = get_update_null_variables(df, exclude_features, target)

df, features, cat_features = set_dummies(

    df = df,

    variables = ['primary_use'],

    cat_features = cat_features,

    features = features

)



df = set_predict_null_values(

    df = df,

    features = features,

    exclude_features = exclude_features,

    features_for_predict = null_variables,

    type_predict = 'Regressor',

    n_estimators=60,

    n_jobs=40,

)



metadata = df.copy()
df = weather_train.copy()



df = pd.merge(

    df,

    pd.merge(

        train, metadata, on = ['building_id'], how = 'left'

    )[['site_id', 'timestamp']].drop_duplicates(),

    on = ['site_id', 'timestamp'],

    how = 'outer'

)



df.sort_values(['timestamp','site_id'], inplace=True)



exclude_features = []

df['Month'] = df['timestamp'].dt.month.astype("uint8")

df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

df['Hour'] = df['timestamp'].dt.hour.astype("uint8")



features = list(df.columns)

null_variables = get_update_null_variables(df, exclude_features, target)



df = set_predict_null_values(

    df = df,

    features = features,

    exclude_features = exclude_features,

    features_for_predict = null_variables,

    type_predict = 'Regressor',

    n_estimators=60,

    n_jobs=40,

)



weather_train = df.copy()
# save test copy

weather_test_check = weather_test.copy()
df = weather_test.copy()

df = pd.merge(

    df,

    pd.merge(

        test,

        metadata,

        on = ['building_id'],

        how = 'left'

    )[['site_id', 'timestamp']].drop_duplicates(),

    on = ['site_id', 'timestamp'],

    how = 'outer'

)



df.sort_values(['timestamp','site_id'], inplace=True)



exclude_features = []



df['Month'] = df['timestamp'].dt.month.astype("uint8")

df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

features = list(df.columns)

null_variables = get_update_null_variables(df, exclude_features, target)



df = set_predict_null_values(

    df = df,

    features = features,

    exclude_features = exclude_features,

    features_for_predict = null_variables,

    type_predict = 'Regressor',

    n_estimators=60,

    n_jobs=40,

)



weather_test = df.copy()
# weather_train

update_feat = ['air_temperature', 'dew_temperature', 'wind_speed', 'wind_direction', 'sea_level_pressure', 'precip_depth_1_hr', 'cloud_coverage']

df.rename(columns = {f:'pred_'+f for f in update_feat}, inplace=True)
df = pd.merge(df, weather_test_check, on = ['site_id', 'timestamp'], how='outer')
# measures = 'dew_temperature'

for meas in update_feat:

    

    df[

        (df['site_id']==7) &\

        (df['Month']==4)

    ][['pred_'+meas, meas]].reset_index(drop=True).plot()

    plt.show();
weather_train.drop(['Month', 'DayOfMonth', 'DayOfWeek','Hour'], axis=1, inplace=True)

weather_test.drop(['Month', 'DayOfMonth', 'DayOfWeek','Hour'], axis=1, inplace=True)
# Not enough memory for commit, so commented out a small number of time features.



from tqdm import tqdm_notebook

for df in tqdm_notebook([train, test]):

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

#     df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

#     df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

#     df['is_year_start'] = df['timestamp'].dt.is_year_start.astype("uint8")

#     df['is_year_end'] = df['timestamp'].dt.is_year_end.astype("uint8")

#     df['weekofyear'] = df['timestamp'].dt.weekofyear.astype("uint8")

#     df['is_month_end'] = df['timestamp'].dt.is_month_end.astype("uint8")

#     df['is_month_start'] = df['timestamp'].dt.is_month_start.astype("uint8")

#     df['dayofyear'] = df['timestamp'].dt.dayofyear.astype("uint16")
train = pd.get_dummies(train, columns = ['meter'])

test = pd.get_dummies(test, columns = ['meter'])
train.head()
weather_train.head()
metadata.head()
cols_float32 = ['square_feet', 'year_built', 'floor_count'] 

weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

metadata[cols_float32] = metadata[cols_float32].astype('float32')

for df in tqdm_notebook([weather_train, weather_test]):

    df[weather_cols] = df[weather_cols].astype('float32')
gc.collect();
print(train.shape)

train = pd.merge(train, metadata, on = ['building_id'], how = 'left')

train = pd.merge(train, weather_train, on = ['site_id', 'timestamp'], how = 'left')

print(train.shape)



del weather_train;

gc.collect();
train.info()
print(test.shape)

test = pd.merge(test, metadata, on = ['building_id'], how = 'left')

test = pd.merge(test, weather_test, on = ['site_id', 'timestamp'], how = 'left')

print(test.shape)



del weather_test;

gc.collect();
train.isnull().sum()
test.isnull().sum()
# pd.concat([train, test], axis=0, sort=False)[

#     [

#         'Month'

#         , 'DayOfMonth'

# #         , 'DayOfWeek'

# #         , 'Hour'

# #         , 'weekofyear'

# #         , 'dayofyear'

#         , 'year_built'

#     ]

# ].nunique()
def get_cyclical_encode(

    df,

    cols_maxval = {},

    is_drop = False

):

    df = df.copy()

    for col in tqdm_notebook(cols_maxval.keys()):

        print('Start ', col)

        df[col + '_sin'] = (np.sin(2 * np.pi * df[col]/cols_maxval[col])).astype('float16')

        df[col + '_cos'] = (np.cos(2 * np.pi * df[col]/cols_maxval[col])).astype('float16')

        print('Add', col + '_sin',col + '_cos')



        if is_drop:

            # drop non-cycle features

            df.drop(col, axis=1, inplace=True)

            print('Drop in features')

    return df



# Not enough memory for commit, so commented out a small number of time features.

cols_maxval = {

    'Month':12

    , 'DayOfMonth':31

#     , 'DayOfWeek':7

#     , 'Hour':24

#     , 'weekofyear':53

#     , 'dayofyear':366

    , 'year_built':751

}
train = get_cyclical_encode(train, cols_maxval, is_drop = True)

test = get_cyclical_encode(test, cols_maxval, is_drop = True)
print(train.shape, test.shape)
train.columns
test.columns
train.info()
test.info()
train['meter_reading'] = np.log1p(train['meter_reading'])
# train.to_pickle(PATH+'train_v2.pkl')

# test.to_pickle(PATH+'test_v2.pkl')
gc.collect();