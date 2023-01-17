import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
import gc
from sklearn.preprocessing import LabelEncoder
base_dir = '../m5-forecasting-accuracy/'
train_dir = os.path.join(base_dir, 'sales_train_evaluation.csv')
test_dir = os.path.join(base_dir, 'sample_submission.csv')
calendar_dir = os.path.join(base_dir, 'calendar.csv')
price_dir = os.path.join(base_dir, 'sell_prices.csv')
sub_dir = os.path.join(base_dir, 'sample_submission.csv')
df_train = pd.read_csv(train_dir)
df_test = pd.read_csv(test_dir)
df_calendar = pd.read_csv(calendar_dir)
df_price = pd.read_csv(price_dir)
df_sub = pd.read_csv(sub_dir)
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
def making_train_data(df_train):
    print("processing train data")
    df_train_after = pd.melt(df_train, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='days', value_name='demand')
    df_train_after['days'] = df_train_after['days'].map(lambda x: int(x[2:]))
    df_train_after = df_train_after.drop(['id'], axis=1)
    df_train_after = reduce_mem_usage(df_train_after)
    gc.collect()
    return df_train_after
def making_test_data(df_test):
    print("processing test data")
    df_test['item_id'] = df_test['id'].map(lambda x: x[:-16])
    df_test['dept_id'] = df_test['item_id'].map(lambda x: x[:-4])
    df_test['cat_id'] = df_test['dept_id'].map(lambda x: x[:-2])
    df_test['store_id'] = df_test['id'].map(lambda x: x[-15:-11])
    df_test['state_id'] = df_test['store_id'].map(lambda x: x[:-2])
    df_test['va_or_ev'] = df_test['id'].map(lambda x: x[-10:])
    df_test_val = df_test.loc[df_test['va_or_ev'] == 'validation', :]
    df_test_ev = df_test.loc[df_test['va_or_ev'] == 'evaluation', :]
    df_test_val_after = pd.melt(df_test_val, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')
    df_test_ev_after = pd.melt(df_test_ev, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')
    df_test_after = pd.concat([df_test_val_after, df_test_ev_after])
    df_test_after['days'] = df_test_after['days'].map(lambda x: int(x[1:]))
    df_test_after.loc[df_test_after['va_or_ev']=='evaluation', ['days']] += 28
    df_test_after['days'] += 1913
    df_test_after = df_test_after.drop(['va_or_ev'], axis=1)
    df_test_after = df_test_after.drop(['id'], axis=1)
    df_test_after = reduce_mem_usage(df_test_after)
    return df_test_after
def making_train_test_data(df_train ,df_test):
    df_train = making_train_data(df_train)
    df_test = making_test_data(df_test)
    print("processing train test data")
    max_train_days = df_train['days'].max()
    min_test_days = df_test['days'].min()
    shift_data = 6
    df_test = pd.concat([df_train.loc[max_train_days - 28 * shift_data <= df_train['days'], :], df_test.loc[df_test['days'] > max_train_days, :]]).reset_index(drop=True)
    
    shift_days_set = [28, 30, 32, 34]
    for i in shift_days_set:
        df_train['pos_demand_{}day'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(i))
        df_test['pos_demand_{}day'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(i))
        gc.collect()
        
    rolling_days_set = [7, 14, 28]
    for i in rolling_days_set:
        
        df_train['demand_{}day_mean'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).mean())
        df_train['demand_{}day_max'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).max())
        df_train['demand_{}day_min'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).min())
        
        df_test['demand_{}day_mean'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).mean())
        df_test['demand_{}day_max'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).max())
        df_test['demand_{}day_min'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).min())
        
        
        df_train = reduce_mem_usage(df_train)
        df_test = reduce_mem_usage(df_test)
        gc.collect()
    
    df_test = df_test.loc[df_test['days'] >= min_test_days, :]
    df_test = reduce_mem_usage(df_test)
    gc.collect()

    
    return df_train, df_test
def making_calendar_data(df_calendar):
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    print("processing calendar data")
    df_calendar['days'] = df_calendar['d'].map(lambda x: int(x[2:]))
    event_type = {np.nan: 1, 'Sporting': 2, 'Cultural': 3, 'National': 5, 'Religious': 7}
    df_calendar['event_type_1'] = df_calendar['event_type_1'].map(event_type)
    df_calendar['event_type_2'] = df_calendar['event_type_2'].map(event_type)
    df_calendar['event_type'] = df_calendar['event_type_1'] * df_calendar['event_type_2']
    le = LabelEncoder()
    le.fit(df_calendar['event_type'])
    df_calendar['event_type'] = le.transform(df_calendar['event_type'])
    df_calendar['cal_day'] = pd.to_datetime(df_calendar['date']).dt.day.astype(np.int8)
    df_calendar['cal_week'] = pd.to_datetime(df_calendar['date']).dt.week.astype(np.int8)
    df_calendar = df_calendar.drop(['event_type_1', 'event_type_2', 'event_name_1', 'event_name_2', 'd', 'weekday', 'date'], axis=1)
#     df_calendar['event_type_1day_ago'] = df_calendar['event_type'].shift(1)
#     df_calendar['event_type_1day_after'] = df_calendar['event_type'].shift(-1)
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    return df_calendar
def making_price_data(df_price):
    df_price = reduce_mem_usage(df_price)
    gc.collect()
    print("processing price data")
    
    shift_days_set = [28, 30, 32, 34]
    for i in shift_days_set:
        df_price['pos_price_{}day'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(i))
    gc.collect()
    
#     rolling_days_set = [7, 14, 28]
#     for i in rolling_days_set:
#         df_price['price_{}day_mean'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(28).rolling(i).mean())
#         df_price['price_{}day_std'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(28).rolling(i).std())
#         df_price = reduce_mem_usage(df_price)
#         gc.collect()
    return df_price
def concat_data(df_train, df_test, df_calendar, df_price):
    df_train, df_test = making_train_test_data(df_train ,df_test)
    df_calendar = making_calendar_data(df_calendar)
    df_price = making_price_data(df_price)
    print("concat data")
    df_train = pd.merge(df_train, df_calendar, on='days', how='left')
    df_test = pd.merge(df_test, df_calendar, on='days', how='left')
    df_train = pd.merge(df_train, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_test = pd.merge(df_test, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_train = df_train.drop(['wm_yr_wk'], axis=1)
    df_test = df_test.drop(['wm_yr_wk'], axis=1)
    del df_calendar, df_price
    gc.collect()
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    return df_train, df_test
def labeling_data(df_train, df_test, df_calendar, df_price):
    df_train, df_test = concat_data(df_train, df_test, df_calendar, df_price)
    print("labeling data")
    label_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in label_columns:
        le  = LabelEncoder()
        le.fit(df_train[c])
        df_train[c] = le.transform(df_train[c])
        df_test[c] = le.transform(df_test[c])
        if c != 'item_id':
            print(le.classes_)
    
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    
    return df_train, df_test
gc.collect()
df_train, df_test = labeling_data(df_train, df_test, df_calendar, df_price)
gc.collect()
for c in df_train.columns:
    print(c)
df_train = df_train.fillna(-10)
df_test = df_test.fillna(-10)
df_train
df_test
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import mean_squared_error

def make_wavenet():

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                  kernel_size = 1,
                  padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                            kernel_size = kernel_size,
                            padding = 'same', 
                            activation = 'tanh', 
                            dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                            kernel_size = kernel_size,
                            padding = 'same',
                            activation = 'sigmoid', 
                            dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                      kernel_size = 1,
                      padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x

    inp = Input(shape=(timesteps, n_used_features))

    x = wave_block(inp, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = Flatten()(x)
    x = Dropout(0.2)(x)

    out = Dense(1, activation = 'relu')(x)

    model = Model(inputs=inp, outputs=out)

    # opt = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
    # opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #   opt = tfa.optimizers.SWA(opt)

    model.compile(optimizer=opt, loss='mse')

    return model

def make_gru():
    inp = Input(shape=(timesteps, n_used_features))
    x = GRU(16, activation='relu', return_sequences=True)(inp)
    x = GRU(32, activation='relu', return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(1, activation='relu')(x)
    model = Model(inputs=[inp], outputs=[out])
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(optimizer=optim, loss='mse')
    
    return model

def make_conv():
    inp = Input(shape=(timesteps, n_used_features))
    x = Conv1D(16, 3, activation='relu')(inp)
    x = Conv1D(32, 3, activation='relu')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='relu')(x)
    model = Model(inputs=[inp], outputs=[out])
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model.compile(optimizer=optim, loss='mse')
    
    return model
def scheduler(epoch):
    lr = 0.001
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr / 3
    elif epoch < 30:
        return lr / 10
    elif epoch < 50:
        return lr / 5
    elif epoch < 60:
        return lr / 10
    elif epoch < 70:
        return lr / 50
    else:
        return lr / 100
df_all = pd.concat([df_train, df_test])
def engineer_data(df_i, aborted_time):
    added_features = []
    for c in ['item_id', 'dept_id']:
        m_d = 'mean_demand_groupby_{}'.format(c)
#         s_d = 'std_demand_groupby_{}'.format(c)
        m_p = 'mean_price_groupby_{}'.format(c)
#         s_p = 'std_price_groupby_{}'.format(c)
        added_features.append(m_d)
#         added_features.append(s_d)
        added_features.append(m_p)
#         added_features.append(s_p)
        
        df_i['mean_demand_groupby_{}'.format(c)] = df_i.loc[df_i['days'] < aborted_time, [c, 'demand']].groupby(c).transform('mean')
#         df_i['std_demand_groupby_{}'.format(c)] = df_i.loc[df_i['days'] < aborted_time, [c, 'demand']].groupby(c).transform('std')
        df_i['mean_price_groupby_{}'.format(c)] = df_i.loc[df_i['days'] < aborted_time, [c, 'sell_price']].groupby(c).transform('mean')
#         df_i['std_price_groupby_{}'.format(c)] = df_i.loc[df_i['days'] < aborted_time, [c, 'sell_price']].groupby(c).transform('std')
    return df_i, added_features
def normalize_data(df_i, used_features, aborted_time, used_time):
    print('used_features:\n{}'.format(used_features))
    mask = (aborted_time <= df_i['days']) & (df_i['days'] < used_time)
    mx = df_i[mask].dropna().max()
    mn = df_i[mask].dropna().min()
    df_i.loc[:, used_features] = (df_i.loc[:, used_features] - mn) / (mx - mn)
    df_i = reduce_mem_usage(df_i)
    return df_i
def make_data(df_i_j, item_id):
    train_position = n_train_data_per_item * item_id
    valid_position = n_valid_data_per_item * item_id
    test_position = n_test_data_per_item * item_id
    for e, i in enumerate(range(test_start_index, train_finish_index)):
        if train_start_index <= i < train_finish_index:
            X_train[train_position+n_train_data_per_item-e-1, :, :] = df_i_j.iloc[-timesteps-i: -i].loc[:, used_features].values
            y_train[train_position+n_train_data_per_item-e-1] = df_i_j.iloc[-i]['demand']
            option_train[train_position+n_train_data_per_item-e-1] = df_i_j.iloc[-i][option_features]
        elif valid_start_index <= i < valid_finish_index:
            X_valid[valid_position+n_valid_data_per_item-e-1, :, :] = df_i_j.iloc[-timesteps-i: -i].loc[:, used_features].values
            y_valid[valid_position+n_valid_data_per_item-e-1] = df_i_j.iloc[-i]['demand']
            option_valid[valid_position+n_valid_data_per_item-e-1] = df_i_j.iloc[-i][option_features]
        elif test_start_index <= i < test_finish_index:
            X_test[test_position+n_test_data_per_item-e-1, :, :] = df_i_j.iloc[-timesteps-i: -i].loc[:, used_features].values
            y_test[test_position+n_test_data_per_item-e-1] = df_i_j.iloc[-i]['demand']
            option_test[test_position+n_test_data_per_item-e-1] = df_i_j.iloc[-i][option_features]
        else:
            print('error')
            break
from sklearn.metrics import mean_squared_error
class ScoreCallback(Callback):
    def __init__(self, X_valid, y_valid):
        self.X_valid = X_valid
        self.y_valid = y_valid

    def on_epoch_end(self, epoch, logs):
        y_valid_pred = self.model.predict(self.X_valid)
        print('   RMSE score {:.4f}'.format(np.sqrt(mean_squared_error(self.y_valid, y_valid_pred))))
gc.collect()
from tqdm import tqdm
from tensorflow.keras.models import load_model

drop_features = ['snap_CA', 'snap_TX', 'snap_WI', 'dept_id', 'cat_id', 'state_id', 'cal_day', 'cal_week', 'wday', 'year', 'month', 'event_type']
fundamental_used_features = [x for x in df_all.columns if x not in drop_features]
option_features = ['store_id', 'item_id', 'days']
target_features = ['demand']
n_fundamental_used_features = len(fundamental_used_features)
n_option_features = len(option_features)
store_ids = df_all['store_id'].unique()
item_ids = df_all['item_id'].unique()

test_finish_index = 28 * 2 + 1
test_start_index = 1
valid_finish_index = 28 * 4 + 1
valid_start_index = test_finish_index
train_finish_index = 28 * 16 + 1
train_start_index = valid_finish_index
used_time = 1969 - train_start_index - 1
aborted_time = 1969 - train_finish_index - 1
timesteps = 7
n_items = 3049

df_total_predict = pd.DataFrame(columns=['store_id', 'item_id', 'days', 'demand'])

n_train_data_per_item = train_finish_index - train_start_index
n_valid_data_per_item = valid_finish_index - valid_start_index 
n_test_data_per_item = test_finish_index - test_start_index

n_train_data = n_train_data_per_item * n_items
n_valid_data = n_valid_data_per_item * n_items
n_test_data = n_test_data_per_item * n_items

for i in store_ids:
    print('finish {} / 10'.format(i+1))

    df_i = df_all.loc[df_all['store_id'] == i, :]
    df_i, added_features = engineer_data(df_i, aborted_time)
    used_features = list(set(fundamental_used_features + added_features) - set(option_features) - set(target_features))
    n_used_features = len(used_features)
    df_i = normalize_data(df_i, used_features, aborted_time, used_time)
    
    X_train = np.zeros((n_train_data, timesteps, n_used_features))
    X_valid = np.zeros((n_valid_data, timesteps, n_used_features))
    X_test = np.zeros((n_test_data, timesteps, n_used_features))
    y_train = np.zeros((n_train_data, ))
    y_valid = np.zeros((n_valid_data, ))
    y_test = np.zeros((n_test_data, ))
    option_train = np.zeros((n_train_data, n_option_features))
    option_valid = np.zeros((n_valid_data, n_option_features))
    option_test = np.zeros((n_test_data, n_option_features))
    
    
    for j in tqdm(item_ids):
        df_i_j = df_i.loc[df_i['item_id'] == j, :].drop(drop_features, axis=1).fillna(-1)
        make_data(df_i_j, j)
            
    model = make_wavenet() #you can choose the model (gru, conv, wavenet)
    model.summary()
    lrsc = LearningRateScheduler(scheduler)
    sc = ScoreCallback(X_valid, y_valid)
    mc = ModelCheckpoint('./models/best_model_store{}.h5'.format(i), monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
    model.fit(X_train, y_train, epochs=30, batch_size=1000, validation_data=(X_valid, y_valid), callbacks=[lrsc, sc, mc])
    model = load_model('./models/best_model_store{}.h5'.format(i))
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    
    plt.figure(figsize=(20, 5))
    plt.title('store_{}_valid'.format(i))
    plt.plot(y_valid)
    plt.plot(y_valid_pred)
    
    plt.figure(figsize=(20, 5))
    plt.title('store_{}_test'.format(i))
    plt.plot(y_test_pred)
    
    plt.show()
    
    df_predict = pd.DataFrame()
    df_predict['store_id'] = option_test[:, 0].astype('int8')
    df_predict['item_id'] = option_test[:, 1].astype('int16')
    df_predict['days'] = option_test[:, 2].astype('int16')
    df_predict['demand'] = y_test_pred
    df_total_predict = pd.concat([df_total_predict, df_predict]).reset_index(drop=True)
    del X_train, X_valid, X_test, y_train, y_valid, y_test, option_train, option_valid, option_test, model
    gc.collect()
df_sub_before = df_test.loc[:, ['store_id', 'item_id', 'days']]
df_sub_before= pd.merge(df_sub_before, df_total_predict, left_on=['item_id', 'store_id', 'days'], right_on=['item_id', 'store_id', 'days'], how='left')
df_sub = pd.read_csv(sub_dir)
df_sub_base = pd.read_csv(sub_dir)
def making_submission(df_sub, df_sub_before, df_sub_base):
    df_sub['va_or_ev'] = df_sub['id'].map(lambda x: x[-10:])
    df_sub_val = df_sub.loc[df_sub['va_or_ev'] == 'validation', :]
    df_sub_ev = df_sub.loc[df_sub['va_or_ev'] == 'evaluation', :]
    df_sub_val = df_sub_val.melt(id_vars=['id', 'va_or_ev'], var_name='days', value_name='demand').drop(['va_or_ev'], axis=1)
    df_sub_ev = df_sub_ev.melt(id_vars=['id', 'va_or_ev'], var_name='days', value_name='demand').drop(['va_or_ev'], axis=1)
    num_va = df_sub_val.shape[0]
    num_ev = df_sub_ev.shape[0]
    df_sub_val['demand'] = df_sub_before['demand'][:num_va].values
    df_sub_ev['demand'] = df_sub_before['demand'][num_va:].values
    df_sub_val = df_sub_val.pivot(index='id', columns='days', values='demand').reset_index()
    df_sub_ev = df_sub_ev.pivot(index='id', columns='days', values='demand').reset_index()
    df_sub_after = pd.concat([df_sub_val, df_sub_ev])
    df_sub_columns = ['id'] + ['F{}'.format(i+1) for i in range(28)]
    df_sub = df_sub_after.loc[:, df_sub_columns]
    df_sub.columns = df_sub_columns
    df_sub = pd.merge(df_sub_base['id'], df_sub, on='id', how='left')
    return df_sub
df_sub = making_submission(df_sub, df_sub_before, df_sub_base)
df_sub.to_csv('./my_GRU_submission.csv', index=False)
df_sub