# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import gc, math



from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb

from tqdm import tqdm

from lightgbm import plot_importance



from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
## Function to reduce the memory usage

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)
# Convert timestamp fiels to datetime64



def convert_to_datetime(df):

    return df['timestamp'].astype('datetime64[ns]')





# Function for extraction timestamp fields such as Year,Month,dat,hour etc...



def extract_timestamp_fields(df):

    

    df['year'] = df['timestamp'].dt.year.astype(np.uint16)

    df['month'] = df['timestamp'].dt.month.astype(np.uint8)

    df['hour'] = df['timestamp'].dt.hour.astype(np.uint8)

    df['day'] = df['timestamp'].dt.day.astype(np.uint8)

    df['building_age'] = df['timestamp'].dt.year.astype(np.uint16) - df.year_built

    return df





# Cyclic Catagorical Encoding for features month,day,hour,wind direcrtion



def cyclic_encoder(df,col):

    if col == 'month':

        df['sine_' + col] = np.sin(2 * np.pi * (df[col] -1)/max(df[col]))

        df['cos_' + col] =  np.cos(2 * np.pi * (df[col] -1)/max(df[col]))

    elif col == 'hour':

        df['sine_' + col] = np.sin(2 * np.pi * df[col]/24)

        df['cos_' + col] =  np.cos(2 * np.pi * df[col]/24)

    elif col == 'day':

        df['sine_' + col] = np.sin(2 * np.pi * df[col]/max(df[col]))

        df['cos_' + col] = np.cos(2 * np.pi * df[col]/max(df[col]))

    else:

        df['sine_' + col] = np.sin(2 * np.pi * df[col]/360.0)

        df['cos_' + col] = np.cos(2 * np.pi * df[col]/360.0)

    return df
weather_train_df['timestamp'] = convert_to_datetime(weather_train_df)

weather_test_df['timestamp'] = convert_to_datetime(weather_test_df)

#Align  timestamp Thanks to Original Author

# Since site_id is in different time zone w.r.t weather conditions ,therefore we have to align the timestamp on site_id



weather = pd.concat([weather_train_df,weather_test_df],ignore_index=True)

weather_key = ['site_id','timestamp']



temp_skeleton = weather[weather_key +['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

#temp_skeleton.head()

temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id',temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

df_2d = temp_skeleton.groupby(['site_id',temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

site_id_offsets = pd.Series(df_2d.values.argmax(axis=1)-14) # temp peak at every 14 hours

site_id_offsets.index.name = 'site_id'

site_id_offsets.head()
# Function 

def timestamp_align(df):

    df['offset'] = df.site_id.map(site_id_offsets)

    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset,unit='H'))

    df['timestamp'] = df['timestamp_aligned']

    del df['timestamp_aligned']

    return df

    
weather_train_df = timestamp_align(weather_train_df)

weather_test_df  = timestamp_align(weather_test_df)

del weather

del df_2d

del temp_skeleton

del site_id_offsets

gc.collect()
%%time

metadata_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')



metadata_df = reduce_mem_usage(metadata_df)

train_df = reduce_mem_usage(train_df)



train_df['timestamp'] = convert_to_datetime(train_df)



train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])

metadata_df['square_feet_log'] = np.log(metadata_df['square_feet'])



# converting primary_use of metadata_df



le = LabelEncoder()

metadata_df['primary_use'] = le.fit_transform(metadata_df['primary_use'])

#metadata_df.head()
# Merge Train DF with metadata and weather train DF's

train_df = train_df.merge(metadata_df,on=['building_id'],how='left')

train_df_final = train_df.merge(weather_train_df,on=['site_id','timestamp'],how='left')

train_df_final.head()
train_df_final.isnull().sum()
missing_value_cols =['year_built','floor_count',

                     'air_temperature','cloud_coverage',

                     'dew_temperature','precip_depth_1_hr',

                     'sea_level_pressure','wind_direction',

                     'wind_speed']



def impute_missing_value(df,missing_value_cols):

    for c in missing_value_cols:

        df[c].fillna(df[c].median(),inplace=True)

print("Imputing Training data missing column values ....")



impute_missing_value(train_df_final,missing_value_cols)

print("Done!...")
train_df_final = extract_timestamp_fields(train_df_final)
del train_df

del weather_train_df

gc.collect()
train_df_final.columns.tolist()
#Calling cyclic_encoder for month,day,hour and wind_direction as these features are cyclic in nature



train_df_final = cyclic_encoder(train_df_final,'month')

train_df_final = cyclic_encoder(train_df_final,'day')

train_df_final = cyclic_encoder(train_df_final,'hour')

train_df_final = cyclic_encoder(train_df_final,'wind_direction')

train_df_final.head()
# # Cyclic Categorical encoding for Test DF  Later run this 

# test_df_final = cyclic_encoder(test_df_final,'month')

# test_df_final = cyclic_encoder(test_df_final,'day')

# test_df_final = cyclic_encoder(test_df_final,'hour')

# test_df_final = cyclic_encoder(test_df_final,'wind_direction')

# test_df_final.head()

# feature_cols = ['building_id', 'meter', 

#        'site_id', 'primary_use', 'square_feet_log',

#        'year_built', 'floor_count', 'air_temperature', 'cloud_coverage',

#        'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

#        'wind_speed', 'year', 'sine_month','cos_month', 'sine_hour',

#         'cos_hour', 'sine_wind_direction','cos_wind_direction',

#         'sine_day', 'cos_day', 'building_age']



# X_train = train_df_final[feature_cols]

# y_train = train_df_final['meter_reading_log1p']



# X_train.head(),y_train.head()

# import lightgbm as lgb

# from tqdm import tqdm



# from sklearn.model_selection import train_test_split

# print("Spliting train and test/validation set...\n")



# train_x,valid_x,train_y,valid_y = train_test_split(X_train,y_train,test_size=0.25,random_state=42)



# print("Trainin Size :",train_x.shape,train_y.shape,"\n")

# print("Test/Validation Size:",valid_x.shape,valid_y.shape)



# print("Splitting Done...")

# categorical_feats = ['building_id','site_id','meter','primary_use',

#                      'sine_month','cos_month', 'sine_hour',

#                      'cos_hour', 'sine_wind_direction',

#                      'cos_wind_direction','sine_day', 'cos_day',]



# # lgbm params

# params = {  'boosting_type': 'gbdt',

#             'objective': 'regression',

#             'metric': {'rmse'},

#             'subsample': 0.25,

#             'subsample_freq': 1,

#             'learning_rate': 0.01,

#             'num_leaves': 31,

#             'feature_fraction': 0.9,

#             'lambda_l1': 1,

#             'lambda_l2': 1

#             }



# #lgb Dataset

# print("Preparing LGB dataset\n")

# d_train = lgb.Dataset(train_x,label=train_y,categorical_feature=categorical_feats)

# d_valid = lgb.Dataset(valid_x,label=valid_y,categorical_feature=categorical_feats)

# valid_sets = [d_train,d_valid]

    

# print("Training LGB ...")

# lgb_reg = lgb.train(params,

#                     train_set=d_train,

#                     num_boost_round=200,

#                     valid_sets=valid_sets,

#                     verbose_eval=20,

#                     early_stopping_rounds=50)

    

# print("LGB Training Done!...")


# fig,ax = plt.subplots(figsize=(12,8))

# plot_importance(lgb_reg,ax=ax)
feature_cols = ['building_id', 'meter', 

       'site_id', 'primary_use', 'square_feet_log',

       'year_built', 'floor_count', 'air_temperature', 'cloud_coverage',

       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

       'wind_speed', 'sine_month','cos_month', 'sine_hour',

        'cos_hour', 'sine_wind_direction','cos_wind_direction',

        'sine_day', 'cos_day', 'building_age']





X_train = train_df_final[feature_cols]

y_train = train_df_final['meter_reading_log1p']
print("Shape of X_train and y_train: ",X_train.shape,y_train.shape)
del train_df_final

gc.collect()
# Let me apply KFold of 5 splits

folds = 5

seed = 555

kf = KFold(n_splits=folds,shuffle=False,random_state=seed)

# lgbm params

params = {  'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.25,

            'subsample_freq': 1,

            'learning_rate': 0.01,

            'num_leaves': 31,

            'feature_fraction': 0.9,

            'lambda_l1': 1,

            'lambda_l2': 1

            }



#X_train[feature_cols].head()
# Let me split the data and train using lightGBM



categorical_feats = ['building_id','site_id','meter','primary_use','sine_month',

                     'cos_month', 'sine_hour','cos_hour', 'sine_wind_direction',

                     'cos_wind_direction','sine_day', 'cos_day']

models = []

for train_idx,valid_idx in kf.split(X_train[feature_cols]):

    

    xtrain = X_train.iloc[train_idx]

    ytrain = y_train.iloc[train_idx]

    xvalid = X_train.iloc[valid_idx]

    yvalid = y_train.iloc[valid_idx]

    

    print("xtrain shape:",xtrain.shape)

    print("xvalid shape:",xvalid.shape)

    print("ytrain shape:",ytrain.shape)

    print("yvalid shape:",yvalid.shape)

    

    #lgb Dataset 

    d_train = lgb.Dataset(xtrain,label=ytrain,categorical_feature=categorical_feats)

    d_valid = lgb.Dataset(xvalid,label=yvalid,categorical_feature=categorical_feats)

    valid_sets = [d_train,d_valid]

    

    print("Training LGB ...")

    lgb_reg1 = lgb.train(params,

                     train_set=d_train,

                     num_boost_round=400,

                     valid_sets=valid_sets,

                     verbose_eval=20,

                     early_stopping_rounds=100)

    models.append(lgb_reg1)

    

    




test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

test_df = reduce_mem_usage(test_df)



print("Converting timestamp to Datetime type .. ")

test_df['timestamp'] = convert_to_datetime(test_df)

print("Size of test_df:",test_df.shape)

print("Merging Test Data sets step 1....")

test_df_final = test_df.merge(metadata_df,on=['building_id'],how='left')



print("Deleting test_df ..")

del test_df

gc.collect()

print("Merging Test Data sets step 2 ....")

test_df_final = test_df_final.merge(weather_test_df,on=['site_id','timestamp'],how='left')

#test_df_final.head()







print("Missing value imputation ...")

impute_missing_value(test_df_final,missing_value_cols)



print("Extracting timestamp into year,month,day and hour ...")

test_df_final = extract_timestamp_fields(test_df_final)



test_df_final = reduce_mem_usage(test_df_final)



print("cyclic encoding testing df...")

test_df_final = cyclic_encoder(test_df_final,'month')

test_df_final = cyclic_encoder(test_df_final,'day')

test_df_final = cyclic_encoder(test_df_final,'hour')

test_df_final = cyclic_encoder(test_df_final,'wind_direction')



test_df_final = reduce_mem_usage(test_df_final)



print("Size of test_df_final:",test_df_final.shape)
print("Deleting test_df_final,weater_test_df and metadata_df ...")



del metadata_df

del weather_test_df

del X_train

#del y_train

gc.collect()
i=0

result=[]

step_size=50000

for _ in tqdm(range(int(np.ceil(test_df_final.shape[0]/step_size)))):

    result.append(np.expm1(sum([model.predict(test_df_final.loc[i:i+step_size-1,feature_cols],num_iteration=model.best_iteration) for model in models])/folds))

    i+=step_size
del test_df_final

gc.collect()
sub_result = np.concatenate(result)

len(result)
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')



submission.shape



submission = reduce_mem_usage(submission)
submission['meter_reading'] = np.clip(sub_result,0,a_max=None)

submission.tail()
submission.to_csv('submission.csv',index=False)