# import numpy as np 

# import pandas as pd 

# import warnings

# warnings.filterwarnings('ignore') # Suppress warnings 

# import gc



# import matplotlib.pyplot as plt

# %matplotlib inline

# import seaborn as sns



# pd.set_option('max_columns', 100)

# pd.set_option('display.float_format', '{:.2f}'.format)



# import os,random, math, psutil, pickle



# from sklearn.metrics import mean_squared_error

# from tqdm import tqdm



# from fastai.tabular import *

# import torch

# print(torch.cuda.is_available())



# # %% [code]

# root = '../input/ashrae-energy-prediction/'

# train_df = pd.read_csv(root + 'train.csv')

# train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



# weather_train_df = pd.read_csv(root + 'weather_train.csv')

# test_df = pd.read_csv(root + 'test.csv')

# weather_test_df = pd.read_csv(root + 'weather_test.csv')

# building_meta_df = pd.read_csv(root + 'building_metadata.csv')

# sample_submission = pd.read_csv(root + 'sample_submission.csv')



# # %% [code]

# print(train_df.shape)

# print(test_df.shape)



# # %% [markdown]

# # **Important: fastai does not work with float16 format unless explicitly specified; thus the below function was modified to exclude float16**



# # %% [code]

# ## Function to reduce the DF size

# def reduce_mem_usage(df, verbose=True):

#     numerics = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']

#     start_mem = df.memory_usage().sum() / 1024**2    

#     for col in df.columns:

#         col_type = df[col].dtypes

#         if col_type in numerics:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)    

#     end_mem = df.memory_usage().sum() / 1024**2

#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

#     return df



# # %% [code]

# ## REducing memory

# train_df = reduce_mem_usage(train_df)

# test_df = reduce_mem_usage(test_df)



# weather_train_df = reduce_mem_usage(weather_train_df)

# weather_test_df = reduce_mem_usage(weather_test_df)

# building_meta_df = reduce_mem_usage(building_meta_df)



# # %% [code]

# # creating the index of test_df to be used for inference

# range_test = np.array_split(test_df.index, 5)

# print(range_test)



# # %% [code]

# train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

# test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])



# weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

# weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])



# building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')



# temp_df = train_df[['building_id']]

# temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

# del temp_df['building_id']

# train_df = pd.concat([train_df, temp_df], axis=1)



# temp_df = test_df[['building_id']]

# temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')



# del temp_df['building_id']

# test_df = pd.concat([test_df, temp_df], axis=1)

# del temp_df, building_meta_df



# temp_df = train_df[['site_id','timestamp']]

# temp_df = temp_df.merge(weather_train_df, on=['site_id','timestamp'], how='left')



# del temp_df['site_id'], temp_df['timestamp']

# train_df = pd.concat([train_df, temp_df], axis=1)



# temp_df = test_df[['site_id','timestamp']]

# temp_df = temp_df.merge(weather_test_df, on=['site_id','timestamp'], how='left')



# del temp_df['site_id'], temp_df['timestamp']

# test_df = pd.concat([test_df, temp_df], axis=1)



# del temp_df, weather_train_df, weather_test_df



# gc.collect()



# # %% [code]

# # Adding small value to meeter reading to avoid log(0) error

# train_df.loc[train_df.meter_reading == 0, ['meter_reading']] = train_df.meter_reading + 0.000001



# # %% [markdown]

# # **Converting variables to log values because of their uneven distribution**



# # %% [code]

# train_df.meter_reading = np.log1p(train_df["meter_reading"])

# train_df.square_feet = np.log1p(train_df["square_feet"])

# test_df.square_feet = np.log1p(test_df["square_feet"])

# train_df['square_feet'] = train_df['square_feet'].astype('float32')

# test_df['square_feet'] = test_df['square_feet'].astype('float32')



# # %% [code]

# # sns.distplot((train_df.meter_reading))

# # sns.distplot(train_df.square_feet)



# # %% [markdown]

# # **Handling missing values**



# # %% [code]

# # # Missing items

# # (train_df.isnull().sum()/ len(train_df) *100).sort_values(ascending = False).head(10)



# # %% [code]

# # # Missing items

# # (test_df.isnull().sum()/ len(test_df) *100).sort_values(ascending = False).head(10)



# # %% [code]

# def average_imputation(df, column_name):

#     imputation = df.groupby(['timestamp'])[column_name].mean()

    

#     df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

#     del imputation

#     return df



# # %% [code]

# train_df = average_imputation(train_df, 'wind_speed')

# #train_df = average_imputation(train_df, 'wind_direction')

# test_df = average_imputation(test_df, 'wind_speed')

# #test_df = average_imputation(test_df, 'wind_direction')



# # %% [code]

# beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

#           (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



# # %% [code]

# for item in beaufort:

#     train_df.loc[(train_df['wind_speed']>=item[1]) & (train_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

# for item in beaufort:

#     test_df.loc[(test_df['wind_speed']>=item[1]) & (test_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]



# # %% [code]

# # def degToCompass(num):

# #     val=int((num/22.5)+.5)

# #     arr=[i for i in range(0,16)]

# #     return arr[(val % 16)]



# # %% [code]

# #train_df['wind_direction'] = train_df['wind_direction'].apply(degToCompass)

# train_df['beaufort_scale'] = train_df['beaufort_scale'].astype('int8')

# #train_df["wind_direction"] = train_df['wind_direction'].astype(np.uint8)



# # %% [code]

# #test_df['wind_direction'] = test_df['wind_direction'].apply(degToCompass)

# test_df['beaufort_scale'] = test_df['beaufort_scale'].astype('int8')

# #test_df["wind_direction"] = test_df['wind_direction'].astype(np.uint8)



# # %% [code]

# def fill_miss(df):

#     miss_col = ['floor_count','year_built','cloud_coverage']

#     for col in miss_col:

#         df[col].fillna(-999, inplace = True)



# # %% [code]

# fill_miss(train_df)

# fill_miss(test_df)



# # %% [code]

# def impute_miss(df):

#     miss_col = ['precip_depth_1_hr', 'sea_level_pressure',  'dew_temperature', 'air_temperature']

#     for col in miss_col:

#         df[col].fillna(df[col].mean(), inplace = True)



# # %% [code]

# impute_miss(train_df)

# impute_miss(test_df)



# # %% [markdown]

# # convert columns with previously missing values to more efficient formats



# # %% [code]

# convert = ['year_built', 'floor_count', 'cloud_coverage', 'precip_depth_1_hr']

# def type_convert(df, columns = convert):

#     for col in columns:

#         df[col] = df[col].astype('int8')



# # %% [code]

# type_convert(train_df)

# type_convert(test_df)



# # %% [code]

# # train_df['month_datetime'] = train_df['timestamp'].dt.month.astype(np.int8)

# # train_df['weekofyear_datetime'] = train_df['timestamp'].dt.weekofyear.astype(np.int8)

# # train_df['dayofyear_datetime'] = train_df['timestamp'].dt.dayofyear.astype(np.int16)

# train_df['hour_datetime'] = train_df['timestamp'].dt.hour.astype('int8')  

# train_df['day_week'] = train_df['timestamp'].dt.dayofweek.astype('int8')

# # train_df['day_month_datetime'] = train_df['timestamp'].dt.day.astype(np.int8)

# # train_df['week_month_datetime'] = train_df['timestamp'].dt.day/7

# # train_df['week_month_datetime'] = train_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    

# # test_df['month_datetime'] = test_df['timestamp'].dt.month.astype(np.int8)

# # test_df['weekofyear_datetime'] = test_df['timestamp'].dt.weekofyear.astype(np.int8)

# # test_df['dayofyear_datetime'] = test_df['timestamp'].dt.dayofyear.astype(np.int16)

# test_df['hour_datetime'] = test_df['timestamp'].dt.hour.astype('int8')

# test_df['day_week'] = test_df['timestamp'].dt.dayofweek.astype('int8')

# # test_df['day_month_datetime'] = test_df['timestamp'].dt.day.astype(np.int8)

# # test_df['week_month_datetime'] = test_df['timestamp'].dt.day/7

# # test_df['week_month_datetime'] = test_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)



# # %% [code]

# train_df.drop(columns = ['timestamp', 'wind_speed'], inplace = True)

# test_df.drop(columns = ['timestamp', 'wind_speed'], inplace = True)



# # %% [code]

# test_cols = ["row_id","site_id", "building_id", "primary_use", "hour_datetime", "day_week",  "meter", 

#              "square_feet", "year_built", "air_temperature", "cloud_coverage",

#               "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']



# # %% [code]

# test_df = test_df[test_cols]



# # %% [code]

# train_cols = ["site_id", "building_id", "primary_use", "hour_datetime", "day_week",  "meter", 

#              "square_feet", "year_built", "air_temperature", "cloud_coverage",

#               "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale', "meter_reading"]



# # %% [code]

# train_df = train_df[train_cols]



# # %% [code]

# train_df.to_pickle('train_df.pkl')

# test_df.to_pickle('test_df.pkl')

# del train_df, test_df

# gc.collect()



# # %% [markdown]

# # **Stage 2 - Load pickled data and train the model**



# # %% [code]

# train_df = pd.read_pickle('train_df.pkl')

# #test_df = pd.read_pickle('test_df.pkl')

# #train_df = train_df[0:100000]

# #test_df = test_df[0:1500000]

# gc.collect()



# # %% [code]

# # #cat_names = list(train_df.select_dtypes(include = ['category', 'bool']).columns)

# # cat_names = ['primary_use', 'meter', 'building_id', 'site_id', 

# #              'month_datetime', 'weekofyear_datetime','dayofyear_datetime',

# #              'hour_datetime','day_week',  'day_month_datetime',

# #              'week_month_datetime', 'floor_count','year_built', 'beaufort_scale']

# cat_names = ["site_id", "building_id", "primary_use", "hour_datetime", "day_week",  "meter"]

# print(cat_names)



# # %% [code]

# # #cont_names = list(train_df.select_dtypes(exclude = ['category', 'bool', 'datetime64[ns]']).columns)

# # #cont_names.remove('meter_reading')

# # cont_names = ['square_feet', 'air_temperature',  'dew_temperature', 'cloud_coverage',

# #               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction' ]





# cont_names = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

#               "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']



# print(cont_names)



# # %% [code]

# #Path / default location for saving/loading models

# path = ''



# #The dependent variable/target

# dep_var = 'meter_reading'



# # %% [code]

# procs = [FillMissing, Categorify, Normalize]



# # %% [code]

# # #Start index for creating a validation set from train_data

# # start_indx = len(train_df) - int(len(train_df) * 0.2)



# # #End index for creating a validation set from train_data

# # end_indx = len(train_df)



# # valid_idx = range(start_indx, end_indx)



# # %% [code]

# data = (TabularList.from_df(train_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

#                 #.split_by_idx(valid_idx)

#                 .split_by_rand_pct(valid_pct = 0.2)

#                 .label_from_df(cols=dep_var, label_cls=FloatList, log = False)

#                 #.add_test(TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names))

#                 #.add_test(test)

#                 .databunch())



# # %% [code]

# #max_log_y = np.log(np.max(train_df['meter_reading'])*1.2)

# #y_range = torch.tensor([0, max_log_y])

# max_y = (np.max(train_df['meter_reading'])*1.2)

# y_range = torch.tensor([0, max_y])



# # %% [code]

# del train_df

# gc.collect()



# # %% [code]

# data.show_batch(rows=5)



# # %% [code]

# data.show_batch(rows=5, ds_type=DatasetType.Valid)



# # %% [code]

# #data.show_batch(rows=5, ds_type=DatasetType.Test)



# # %% [code]

# learn = tabular_learner(data, layers=[800,400], ps=[0.001,0.01], emb_drop=0.04, y_range = y_range, emb_szs={'building_id': 50}, metrics= rmse)

# #metrics= exp_rmspe



# # %% [code]

# learn.model



# # %% [code]

# learn.lr_find()



# # %% [code]

# learn.recorder.plot()



# # %% [code]

# gc.collect()



# # %% [code]

# learn.fit_one_cycle(1, max_lr = 3e-1, wd = 0.2)



# # %% [code]

# learn.recorder.plot_losses()



# # %% [code]

# learn.export()



# # %% [code]

# learn.destroy()



# # %% [markdown]

# # **Stage 3 - Batch inference from the test dataset**



# # %% [code]

# output = []

# for r in range(len(range_test)):

#     test_df = pd.read_pickle('test_df.pkl')

#     test_df = test_df.ix[range_test[r]]

#     test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)

#     del test_df

#     gc.collect()

#     # beginning of inference

#     learn = load_learner(".", test= test)

#     preds,_ = learn.get_preds(ds_type=DatasetType.Test)

#     preds = np.expm1(preds.numpy())

#     preds = pd.DataFrame(preds)

#     output.append(preds)   



# # %% [code]

# meter_reading = pd.concat(output, ignore_index = True)



# # %% [code]

# meter_reading.reset_index(inplace = True)



# # %% [code]

# meter_reading.columns = ['row_id', 'meter_reading']



# # %% [code]

# meter_reading.head()



# # %% [code]

# meter_reading.to_csv('submission.csv', index=False)



# # %% [code]

# # submission = pd.DataFrame()

# # submission['row_id'] = row_id

# # submission['meter_reading'] = output

# # submission.head()

# # submission.to_csv('submission.csv', index=False)