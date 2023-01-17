timesteps = 28
startDay = 0
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import scipy as sc
import gc #importing garbage collector
import time
import sys
from scipy import signal
from itertools import chain


import warnings
warnings.filterwarnings('ignore')

%matplotlib inline  

SEED = 42
#Pandas - Displaying more rorws and columns
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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
#!kaggle competitions download -c m5-forecasting-accuracy
df_train = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
#df_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
df_days = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

df_train = reduce_mem_usage(df_train)
#df_prices = reduce_mem_usage(df_prices)
df_days = reduce_mem_usage(df_days)
df_train=df_train.T
df_train.iloc[:10,:10]
df_train.shape
df_days.tail(10)
df_days.shape
#my_df_summary(df_train)
#my_df_summary(df_prices)
df_days['is_workday'] = 0
df_days['is_workday'].loc[df_days['wday']>2] =1
df_days['is_workday'] = df_days['is_workday'].astype(np.int8)
df_days['is_event_day'] = [1 if x ==False else 0 for x in df_days['event_name_1'].isnull()] 
df_days['is_event_day'] = df_days['is_event_day'].astype(np.int8)
df_days["date"] = pd.to_datetime(df_days['date'])
df_days['week'] = df_days["date"].dt.week
df_days['week'] = df_days['week'].astype(np.int8)
df_days['num_events_week'] = df_days.groupby(by=['year','week'])['is_event_day'].transform('sum')
df_days['num_events_week'] = df_days['num_events_week'].astype(np.int8)
df_days['is_event_week'] = [1 if x >0 else 0 for x in df_days['num_events_week']]
df_days['is_event_week'] = df_days['is_event_week'].astype(np.int8)
df_days.info()
df_days.set_index('date', inplace=True)
day_after_event = df_days[df_days['is_event_day']==1].index.shift(1,freq='D')
df_days['is_event_day_after'] = 0
df_days['is_event_day_after'][df_days.index.isin(day_after_event)] = 1
df_days['is_event_day_after'] = df_days['is_event_day_after'].astype(np.int8)

del day_after_event
day_before_event = df_days[df_days['is_event_day']==1].index.shift(-1,freq='D')
df_days['is_event_day_before'] = 0
df_days['is_event_day_before'][df_days.index.isin(day_before_event)] = 1
df_days['is_event_day_before'] = df_days['is_event_day_before'].astype(np.int8)

del day_before_event
df_days.loc[:, "is_sport_event"] = ((df_days["event_type_1"] == "Sporting") | (df_days["event_type_2"] == "Sporting")).astype("int8")
df_days.loc[:, "is_cultural_event"] = ((df_days["event_type_1"] == "Cultural") | (df_days["event_type_2"] == "Cultural")).astype("int8")
df_days.loc[:, "is_national_event"] = ((df_days["event_type_1"] == "National") | (df_days["event_type_2"] == "National")).astype("int8")
df_days.loc[:, "is_religious_event"] = ((df_days["event_type_1"] == "Religious") | (df_days["event_type_2"] == "Religious")).astype("int8")
# adding 'id' column as well as 'cat_id', 'dept_id' and 'state_id', then changing the type to 'categorical'
#df_prices.loc[:, "id"] = df_prices.loc[:, "item_id"] + "_" + df_prices.loc[:, "store_id"] + "_validation"
#df_prices['state_id'] = df_prices['store_id'].str.split('_',expand=True)[0]
#df_prices = pd.concat([df_prices, df_prices["item_id"].str.split("_", expand=True)], axis=1)
#df_prices = df_prices.rename(columns={0:"cat_id", 1:"dept_id"})
#df_prices[["store_id", "item_id", "cat_id", "dept_id", 'state_id']] = df_prices[["store_id","item_id", "cat_id", "dept_id", 'state_id']].astype("category")
#df_prices = df_prices.drop(columns=2)
#
## It seems that leadings zero values in each train_df item row are not real 0 sales but mean absence for the item in the store
## we can safe some memory by removing such zeros
#release_df = df_prices.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
#release_df.columns = ['store_id','item_id','release']
#
#df_prices = pd.merge(df_prices, release_df, on=['store_id','item_id'])
#del release_df
# We can do some basic aggregations
#df_prices['price_max'] = df_prices.groupby(['store_id','item_id'])['sell_price'].transform('max')
#df_prices['price_min'] = df_prices.groupby(['store_id','item_id'])['sell_price'].transform('min')
#df_prices['price_std'] = df_prices.groupby(['store_id','item_id'])['sell_price'].transform('std')
#df_prices['price_mean'] = df_prices.groupby(['store_id','item_id'])['sell_price'].transform('mean')
#df_prices['price_norm'] = df_prices['sell_price']/df_prices['price_max']
#
#df_prices['price_nunique'] = df_prices.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
#df_prices['item_nunique'] = df_prices.groupby(['store_id','sell_price'])['item_id'].transform('nunique').astype(np.int8)
df_days[['wm_yr_wk','month','year']].head()
# I would like some "rolling" aggregations
# but would like months and years as "window"
#calendar_prices = df_days[['wm_yr_wk','month','year']]
#calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
#df_prices = df_prices.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
#del calendar_prices
#df_prices['price_momentum'] = df_prices['sell_price']/df_prices.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
#df_prices['price_momentum_m'] = df_prices['sell_price']/df_prices.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
#df_prices['price_momentum_y'] = df_prices['sell_price']/df_prices.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')
#df_prices
#df_prices['release'] = df_prices['release'] - df_prices['release'].min()
#df_prices['release'] = df_prices['release'].astype(np.int16)
gc.collect()
df_train_full = df_train.copy()
startDay = 0
timesteps = 14
daysBeforeEventTest = df_days['is_event_day_before'][1941:1969]
daysBeforeEvent = df_days['is_event_day_before'][startDay:1941]
daysBeforeEvent.index = df_train.iloc[6:,:].index[startDay:1941]
df_final = pd.concat([df_train, daysBeforeEvent], axis = 1)
df_final.columns
df_final = df_final[startDay:]
#Feature Scaling
#Scale the features using min-max scaler in range 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
dt_scaled = scaler.fit_transform(df_final.iloc[6:,:])
dt_scaled.shape
X_train = []
y_train = []
for i in range(timesteps, 1941 - startDay):
    X_train.append(dt_scaled[i-timesteps:i])
    y_train.append(dt_scaled[i][0:dt_scaled.shape[1]-1]) 
X_train = np.array(X_train)
y_train = np.array(y_train)
print('Shape of X_train :'+str(X_train.shape))
print('Shape of X_train :'+str(y_train.shape))
gc.collect()
inputs = df_final[-timesteps:]
inputs = scaler.transform(inputs)
df_final[-timesteps:].shape
gc.collect()
from keras import backend as K
# defining rmse as loss function
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
# defining wrmsse as loss function
#%run ./WRMSSE.ipynb
%who
del  df_final, df_train_full, df_days, df_train, dt_scaled, time, sys, signal, reduce_mem_usage
gc.collect()
# Importing the Keras libraries and packages
import tensorflow_probability as tfp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
layer_1_units=40
model.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
layer_2_units=400
model.add(LSTM(units = layer_2_units, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
layer_3_units=400
model.add(LSTM(units = layer_3_units))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = y_train.shape[1]))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = root_mean_squared_error)
# alternative loss 'mse' or wrmsse
# Fitting the RNN to the Training set
epoch_no=100 # going through the dataset 32 times
batch_size_RNN=44 # with each training step the model sees 44 examples
fit = model.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size_RNN)
plt.plot(fit.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
model.layers
gc.collect()
X_test = []
X_test.append(inputs[0:timesteps])
X_test = np.array(X_test)
predictions = []

for j in range(timesteps,timesteps + 28):
    predicted_volume = model.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 30491))
    testInput = np.column_stack((np.array(predicted_volume), daysBeforeEventTest[0 + j - timesteps]))
    X_test = np.append(X_test, testInput).reshape(1,j + 1,30491)
    predicted_volume = scaler.inverse_transform(testInput)[:,0:30490]
    predictions.append(predicted_volume)
submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))

submission = submission.T
    
submission = pd.concat((submission, submission), ignore_index=True)

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
    
idColumn = sample_submission[["id"]]
    
submission[["id"]] = idColumn  

cols = list(submission.columns)
cols = cols[-1:] + cols[:-1]
submission = submission[cols]

colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]

submission.columns = colsdeneme

submission.to_csv("submission_evaluation_newloss_nodrop.csv", index=False)
predictions