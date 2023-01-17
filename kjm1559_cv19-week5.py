# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings(action='ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAIN_DATA_PATH = '/kaggle/input/covid19-global-forecasting-week-5/train.csv'

TEST_DATA_PATH = '/kaggle/input/covid19-global-forecasting-week-5/test.csv'

POPULATION_DATA_PATH = '/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv'

TARGET_DATE = 21
CAL_TYPE = {'County':'category', 'Province_State':'category', 'Country_Region':'category', 'Id': 'int32', \

            'Population':'int64', 'Weight':'float32', 'Date':'str', 'Target':'str'}

CAL_DATA = ['id', 'week', 'day', 'month']

INPUT_DATA = ['target_before', 'target_3']

def make_dataset():

    train_data = pd.read_csv(TRAIN_DATA_PATH, dtype=CAL_TYPE)

    train_data['Date'] = pd.to_datetime(train_data['Date'])

    train_data['week'] = train_data['Date'].dt.weekday

    train_data['day'] = train_data['Date'].dt.day

    train_data['month'] = train_data['Date'].dt.month

    train_data['id'] = train_data['Country_Region'].str.cat(train_data['Province_State'], sep=' ', na_rep='')

    train_data['id'] = train_data['id'].str.cat(train_data['County'], sep=' ', na_rep='')

    train_data['id'] = train_data['id'].astype('category')

    train_data['id'] = train_data['id'].cat.codes.astype('int16')

    c_train_data = train_data[train_data['Target'] == 'ConfirmedCases']

    f_train_data = train_data[train_data['Target'] == 'Fatalities']

    

    c_train_data['target_before'] = c_train_data.groupby(by='id')['TargetValue'].shift(periods=1)

    c_train_data['target_7'] = c_train_data.groupby(by='id')['target_before'].rolling(7).mean().reset_index(0, drop=True)

    c_train_data['target_5'] = c_train_data.groupby(by='id')['target_before'].rolling(5).mean().reset_index(0, drop=True)

    c_train_data['target_3'] = c_train_data.groupby(by='id')['target_before'].rolling(3).mean().reset_index(0, drop=True)

    

    f_train_data['target_before'] = f_train_data.groupby(by='id')['TargetValue'].shift(periods=1)

    f_train_data['target_7'] = f_train_data.groupby(by='id')['target_before'].rolling(7).mean().reset_index(0, drop=True)

    f_train_data['target_5'] = f_train_data.groupby(by='id')['target_before'].rolling(5).mean().reset_index(0, drop=True)

    f_train_data['target_3'] = f_train_data.groupby(by='id')['target_before'].rolling(3).mean().reset_index(0, drop=True)

    

#     train_data.dropna(inplace=True)

#     train_data.dropna(subset=['target_before'], inplace=True)

    f_train_data.dropna(subset=['target_7'], inplace=True)

    c_train_data.dropna(subset=['target_7'], inplace=True)

    return c_train_data, f_train_data

    

def make_X(df, batch_size):

    X = {'inputs': df[INPUT_DATA].to_numpy().reshape((batch_size, TARGET_DATE,3))}

    for i, v in enumerate(CAL_DATA):

            X[v] = df[[v]].to_numpy().reshape((batch_size, TARGET_DATE, 1))

    return X



def make_X2(df):

    X = {'inputs': df[INPUT_DATA].to_numpy()}

    for i, v in enumerate(CAL_DATA):

            X[v] = df[[v]].to_numpy()

    return X
c_train_data, f_train_data = make_dataset()
c_train_data[c_train_data['TargetValue'] < 0].id.unique()
c_train_data[c_train_data['id'] == 11]['target_3'].plot()
c_train_data[c_train_data['id'] == 11]['TargetValue'].plot()
from datetime import datetime

from datetime import timedelta

import tensorflow.keras

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Input, GRU, Masking, Permute, Concatenate, BatchNormalization, Flatten, Embedding, TimeDistributed, Reshape, Dropout, Concatenate

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K

import tensorflow as tf

from tqdm import tqdm

import pandas as pd

import numpy as np

import gzip

import pickle



def pinball_loss(true, pred):

    assert pred.shape[0]==true.shape[0]

    tau = K.constant(np.array([0.05, 0.5, 0.95]))

    loss1 = (true[:, 0:1] - pred) *  tau

    loss2 = (pred - true[:, 0:1]) * (1 - tau)

    print(true[:, 1].shape, K.mean(loss1).shape, pred.shape)

    print((true[:, 1] * K.mean(loss1, axis=1)).shape, loss1[pred <= true[:, 0:1]].shape)

    loss1 = K.mean(true[:, 1] * K.mean(loss1[pred <= true[:, 0:1]]))/3

    loss2 = K.mean(true[:, 1] * K.mean(loss2[pred > true[:, 0:1]]))/3

    return (loss1 + loss2) / 2



def pinball_loss1(true, pred):

    assert pred.shape[0]==true.shape[0]

    loss1 = (true[:, 0:1] - pred) *  0.05

    loss2 = (pred - true[:, 0:1]) * (1 - 0.05)

    loss1 = K.clip(loss1, 0, K.max(loss1))

    loss2 = K.clip(loss2, 0, K.max(loss2))

    loss = loss1 + loss2

    print(loss.shape, pred.shape)

    loss = K.mean(true[:, 1:2] * loss)

    return loss

def pinball_loss2(true, pred):

    assert pred.shape[0]==true.shape[0]

    loss1 = (true[:, 0] - pred)# *  0.5

    loss2 = (pred - true[:, 0])# * (1 - 0.5)

    loss1 = K.clip(loss1, 0, K.max(loss1))

    loss2 = K.clip(loss2, 0, K.max(loss2))

    loss = loss1 + loss2

    loss = K.mean(true[:, 1] * loss)

    return loss

def pinball_loss3(true, pred):

    assert pred.shape[0]==true.shape[0]

    loss1 = (true[:, 0:1] - pred) *  0.95

    loss2 = (pred - true[:, 0:1]) * (1 - 0.95)

    loss1 = K.clip(loss1, 0, K.max(loss1))

    loss2 = K.clip(loss2, 0, K.max(loss2))

    loss = loss1 + loss2

    loss = K.mean(true[:, 1:2] * loss)

    return loss



def rmse(true, pred):

    loss1 = (true[:, 0:1] - pred) 

    loss2 = (pred - true[:, 0:1])

    loss1_005 = loss1 * 0.05

    loss1_005 = K.clip(loss1_005, 0, K.max(loss1_005))

    loss2_005 = loss2 * 0.95

    loss2_005 = K.clip(loss2_005, 0, K.max(loss2_005))

    loss_005 = loss1_005 + loss2_005

    

    loss1_05 = loss1 * 0.5

    loss1_05 = K.clip(loss1_05, 0, K.max(loss1_05))

    loss2_05 = loss2 * 0.5

    loss2_05 = K.clip(loss2_05, 0, K.max(loss2_05))

    loss_05 = loss1_05 + loss2_05

    

    loss1_095 = loss1 * 0.95

    loss1_095 = K.clip(loss1_095, 0, K.max(loss1_095))

    loss2_095 = loss2 * 0.05

    loss2_095 = K.clip(loss2_095, 0, K.max(loss2_095))

    loss_095 = loss1_095 + loss2_095

    

    loss = K.mean(true[:, 1:2] * ((loss_005 + loss_05 + loss_095) / 3))

    

#     loss1 = K.clip(loss1, 0, K.max(loss1))

#     loss2 = K.clip(loss2, 0, K.max(loss2))

    

    return loss

#     return K.mean(K.abs(true[:,0:1] - pred) * true[:, 1:2])





def simple_model(input_size, days=21, batch_size=2**14, epochs=200, lr=1e-3):

    

    inputs = Input(shape=(input_size), name='inputs')    

    

    id_input = Input(shape=(1,), name='id')

    wday_input = Input(shape=(1,), name='week')

    day_input = Input(shape=(1,), name='day')

    month_input = Input(shape=(1,), name='month')

    

    id_emb = Flatten()(Embedding(3464, 3)(id_input))

    wday_emb = Flatten()(Embedding(8, 1)(wday_input))

    day_emb = Flatten()(Embedding(32, 3)(day_input))

    month_emb = Flatten()(Embedding(13, 2)(month_input))

    

#     x = Concatenate(-1)([inputs, id_emb, wday_emb, day_emb, month_emb])

    x = Concatenate(-1)([inputs, id_emb, day_emb, month_emb])



    x = Dense(128, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    x = Dense(32, activation='relu')(x)        

    outputs1 = Dense(1, activation='linear')(x) 

#     outputs2 = Dense(1, activation='linear')(x) 

#     outputs3 = Dense(1, activation='linear')(x) 

#     print(outputs.shape)



#     outputs1 = outputs1 * 46197 - 10034

#     outputs2 = outputs2 * 46197 - 10034

#     outputs3 = outputs3 * 46197 - 10034

    

    input_dic = {

        'inputs': inputs, 'week': wday_input, 'month': month_input, 

        'day': day_input, 'id': id_input,



    }



    optimizer = Adam(lr=lr, name='adam')

    model = Model(input_dic, outputs1, name='gru_network')

    model.compile(optimizer=optimizer, loss=rmse)

#     model.compile(optimizer=optimizer, loss=pinball_loss2)

#     model.compile(optimizer=optimizer, loss=['mse', 'mse', 'mse'])

    return model

# make_ID_list(c_train_data)
# f_model = attention_model(3,)

# c_model = attention_model(3,)

# c_gen = DataGenerator(c_train_data, 2**10, make_ID_list(c_train_data))

# f_gen = DataGenerator(f_train_data, 2**10, make_ID_list(f_train_data))
# b = make_X2(c_train_data)
#Normalization

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

# fitted = min_max_scaler.fit(c_train_data[INPUT_DATA + ['TargetValue']])

# print(fitted.data_max_)

c_train_data_norm = c_train_data.copy()

# c_train_data_norm[INPUT_DATA + ['TargetValue']] = min_max_scaler.transform(c_train_data[INPUT_DATA + ['TargetValue']])

# output = min_max_scaler.transform(output)
c_s_model = simple_model(2,)

model_path = './cv19_predict5.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(patience=10)

hist = c_s_model.fit(make_X2(c_train_data_norm), c_train_data_norm[['TargetValue', 'Weight']].values, batch_size=2**14, epochs=100, validation_split=0.2, callbacks=[cb_checkpoint, early_stopping], shuffle=True)
# c_s_e_model = esemble_model(3,)

# model_path = './e_cv19_predict5.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'

# cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

# early_stopping = EarlyStopping(patience=10)

# hist = c_s_e_model.fit(make_X2(c_train_data_norm), [c_train_data_norm[['TargetValue', 'Weight']].values], batch_size=256, epochs=100, validation_split=0.2, callbacks=[cb_checkpoint, early_stopping], shuffle=True)
c_s_model.load_weights('./cv19_predict5.h5')

test_result = c_s_model.predict(make_X2(c_train_data_norm))

# print(test_result, c_train_data_norm['TargetValue'])
from matplotlib import pyplot as plt

# plt.plot(test_result[0].reshape(349763))

plt.plot(test_result.reshape(len(c_train_data_norm)), alpha=0.5)

# plt.plot(test_result[2].reshape(349763))

plt.plot(c_train_data_norm['TargetValue'].reset_index(drop=True), alpha=0.5)

# plt.ylim(-100, 100)
# plt.plot(c_train_data_norm['TargetValue'])

# plt.show()

plt.plot(c_train_data_norm['TargetValue'].reset_index(drop=True))

# plt.ylim(-100, 100)
min_max_scaler_f = MinMaxScaler()

# fitted = min_max_scaler_f.fit(f_train_data[INPUT_DATA + ['TargetValue']])

# print(fitted.data_max_)

f_train_data_norm = f_train_data.copy()

# f_train_data_norm[INPUT_DATA + ['TargetValue']] = min_max_scaler.transform(f_train_data[INPUT_DATA + ['TargetValue']])

# output = min_max_scaler.transform(output)
f_s_model = simple_model(2,)

model_path = './cv19_predict_f.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(patience=10)

hist = f_s_model.fit(make_X2(f_train_data_norm), [f_train_data_norm[['TargetValue', 'Weight']].values], batch_size=2**14, epochs=100, validation_split=0.2, callbacks=[cb_checkpoint, early_stopping])
c_train_data, f_train_data = make_dataset()
test_data = pd.read_csv(TEST_DATA_PATH, dtype=CAL_TYPE)

test_data['Date'] = pd.to_datetime(test_data['Date'])

test_data['Date'] = pd.to_datetime(test_data['Date'])

test_data['week'] = test_data['Date'].dt.weekday

test_data['day'] = test_data['Date'].dt.day

test_data['month'] = test_data['Date'].dt.month

test_first_date = test_data['Date'].unique()[0]

exist_data_c = c_train_data[c_train_data['Date'] >= test_first_date]

exist_data_f = f_train_data[f_train_data['Date'] >= test_first_date]

# test_data['Target']

target_list = np.zeros(len(test_data))

# test_data['target_before']

target_before_list = np.zeros(len(test_data))

# make id

test_data['id'] = test_data['Country_Region'].str.cat(test_data['Province_State'], sep=' ', na_rep='')

test_data['id'] = test_data['id'].str.cat(test_data['County'], sep=' ', na_rep='')

test_data['id'] = test_data['id'].astype('category')

test_data['id'] = test_data['id'].cat.codes.astype('int16')







c_test_data = test_data[test_data.Target == 'ConfirmedCases']

f_test_data = test_data[test_data.Target == 'Fatalities']

# c_test_data['target_7'] = c_test_data.groupby(by='id')['target_before'].rolling(7).mean().reset_index(0, drop=True)

date_list = c_test_data.sort_values(by = 'Date').Date.unique()
def test_seq(date_list, test_data, test_model, min_max_scaler):

    test_data.loc[test_data.index, 'target_before'] = test_data.groupby(by='id')['TargetValue'].shift(periods=1)

    test_data.loc[test_data.index, 'target_7'] = test_data.groupby(by='id')['target_before'].rolling(7).mean().reset_index(0, drop=True)

    test_data.loc[test_data.index, 'target_3'] = test_data.groupby(by='id')['target_before'].rolling(3).mean().reset_index(0, drop=True)

    group_by_test = test_data.groupby(by='Date')

    result_data = []

    for data in tqdm(date_list, position=0):

        tmp_test_data = group_by_test.get_group(data)

#         if True:

        if np.sum(pd.isnull(tmp_test_data.TargetValue)) != 0:

#             print(tmp_test_data[['target_before', 'target_7', 'target_3']])

#             tmp_test_data.loc[tmp_test_data.index, INPUT_DATA + ['TargetValue']] = min_max_scaler.transform(tmp_test_data[INPUT_DATA + ['TargetValue']])

#             print(make_X2(tmp_test_data))

            pre_data = test_model.predict_on_batch(make_X2(tmp_test_data))

#             print(pre_data)

#             tmp_test_data['TargetValue_1'] = pre_data[0].numpy()

#             tmp_test_data['TargetValue_3'] = pre_data[2].numpy()

            tmp_test_data['TargetValue'] = pre_data.numpy().reshape(len(tmp_test_data))

#             tmp_test_data[INPUT_DATA + ['TargetValue_1']] = min_max_scaler.inverse_transform(tmp_test_data[INPUT_DATA + ['TargetValue_1']])

#             tmp_test_data[INPUT_DATA + ['TargetValue_3']] = min_max_scaler.inverse_transform(tmp_test_data[INPUT_DATA + ['TargetValue_3']])

#             tmp_test_data[INPUT_DATA + ['TargetValue']] = min_max_scaler.inverse_transform(tmp_test_data[INPUT_DATA + ['TargetValue']])

            test_data.loc[tmp_test_data.index, 'TargetValue'] = tmp_test_data['TargetValue']

#             print(tmp_test_data[['TargetValue_1', 'TargetValue_3', 'TargetValue']])

            for idata in tmp_test_data.itertuples():

                result_data.append([str(int(idata.ForecastId)) + '_0.05', idata.TargetValue])

                result_data.append([str(int(idata.ForecastId)) + '_0.5', idata.TargetValue])

                result_data.append([str(int(idata.ForecastId)) + '_0.95', idata.TargetValue])

            test_data = test_data.sort_values(by=['id', 'Date'])

            test_data['target_before'] = test_data.groupby(by='id')['TargetValue'].shift(periods=1)

            test_data['target_7'] = test_data.groupby(by='id')['target_before'].rolling(7).mean().reset_index(0, drop=True)

            test_data['target_5'] = test_data.groupby(by='id')['target_before'].rolling(5).mean().reset_index(0, drop=True)

            test_data['target_3'] = test_data.groupby(by='id')['target_before'].rolling(3).mean().reset_index(0, drop=True)

            group_by_test = test_data.groupby(by='Date')    

            

        else:

            for idata in tmp_test_data.itertuples():

                result_data.append([str(int(idata.ForecastId)) + '_0.05', idata.TargetValue])

                result_data.append([str(int(idata.ForecastId)) + '_0.5', idata.TargetValue])

                result_data.append([str(int(idata.ForecastId)) + '_0.95', idata.TargetValue])

    return pd.DataFrame(result_data, columns=['ForecastId_Quantile', 'TargetValue'])

            

        
tt = pd.merge(c_test_data[['ForecastId', 'County', 'Province_State', 'Country_Region',

       'Population', 'Weight', 'Date', 'Target', 'week', 'day', 'month', 'id',]], c_train_data, how='outer', on=[ 'County', 'Province_State', 'Country_Region', 'Population', 'Weight', 'Date', 'Target', 'week', 'day', 'month', 'id'])

c_s_model.load_weights('./cv19_predict5.h5')

# tt=tt.sort_values(by='Date')

c_result = test_seq(date_list, tt.sort_values(by=['id', 'Date']), c_s_model, min_max_scaler)
c_result
tt = pd.merge(f_test_data[['ForecastId', 'County', 'Province_State', 'Country_Region',

       'Population', 'Weight', 'Date', 'Target', 'week', 'day', 'month', 'id',]], f_train_data, how='outer', on=[ 'County', 'Province_State', 'Country_Region', 'Population', 'Weight', 'Date', 'Target', 'week', 'day', 'month', 'id'])

f_s_model.load_weights('./cv19_predict_f.h5')

# tt=tt.sort_values(by='Date')

f_result = test_seq(date_list, tt.sort_values(by=['id', 'Date']), f_s_model, min_max_scaler_f)
result = pd.concat([c_result, f_result])

result.to_csv('submission.csv', index=False)
result[result['ForecastId_Quantile'] == '207_0.05']
# tt