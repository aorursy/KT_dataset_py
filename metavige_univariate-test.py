# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from datetime import datetime

import os



import numpy as np

import pandas as pd

import sklearn

from keras.layers import Activation, Dense, Dropout, GRU, CuDNNGRU

from pandas import DataFrame, read_csv

import matplotlib.pyplot as plot



from keras import Sequential

from keras.engine.saving import load_model

from keras.callbacks import TensorBoard, EarlyStopping

from keras.optimizers import adam



import tensorflow

# 忽略 tensorflow 的 warning 訊息

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



warnings.filterwarnings('ignore')



EXPORT_DIR = '/kaggle/input'

# Any results you write to the current directory are saved as output



print(tensorflow.__version__)
!nvidia-smi
import numpy as np

from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.preprocessing import StandardScaler





def generate_target_series(input_series, output=1, look_back=1):

    """

    產生多步輸出 target 的序列資料，因為輸出的序列資料，在 TimeseriesGenerator 無法產生多步輸出

    :param input_series: 序列資料

    :param output: 要預測的時間序列長度

    :param look_back: 要回頭看多久的時間區段點，來當作預測的資料，也代表實際要預測的時間點與傳入要用來學習 x 的時間差異

    :return:

    """

    target_series = []

    total_length = len(input_series)



    series_start = look_back - 1

    series_end = total_length - output



    for n in range(total_length):

        if series_start < n <= series_end:

            target_series.append(input_series[n:n + output])

        else:

            if n > series_end:

                data = input_series[n: total_length]

                # 前面補齊 0

                data = np.pad(data, (0, output - len(data)), mode='constant', constant_values=0)

                target_series.append(data)

            else:

                if n <= series_start:

                    target_series.append(np.zeros(output))

    return np.array(target_series)





def get_generator(data, output, time_step, batch_size=1, target=None) -> TimeseriesGenerator:

    """

    將資料轉換成 TimeseriesGenerator 的序列物件

    :param data: 輸入的 x (2 dim)

    :param output:

    :param time_step:

    :param batch_size: 批次處理數量

    :param target: 輸出的 y (2 dim)

    :return:

    """

    if data.shape == 1:

        _data_series = data.reshape(len(data), 1)

    else:

        _data_series = data



    if target is None:

        _target_series = _data_series

    else:

        _target_series = target

    #     print(_target_series)



    # 目前 TimeseriesGenerator 是沒有辦法產生 multi-step 的輸出的，所以要自己產生好輸出

    # 將資料轉換成 [ samples, features ]

    _target_series = generate_target_series(_target_series,

                                            output=output,

                                            look_back=time_step)

    #     print('target: {}'.format(_target_series.shape))



    return TimeseriesGenerator(data=_data_series,

                               targets=_target_series,

                               length=time_step,

                               batch_size=batch_size)





class TimeSeriesDataProcessor:

    """

    處理要訓練的資料

    """



    def __init__(self, fit_data, scalar=None):

        if scalar is None:

            scalar = StandardScaler()

        self.__scalar = scalar

        self.__fit_data = fit_data



        self.__scalar.fit(fit_data)



    def transform(self, data=None):

        if data is None:

            data = self.__fit_data



        return self.__scalar.transform(data)
def resample_data(df: DataFrame):

    """

    將整天的資料填滿每分鐘的時間點

    :param df:

    :return:

    """

    num_idx = 1440

    # 用一個空白 INDEX 的 DataFrame，可以填滿時間點

    idx_df = DataFrame(index=pd.date_range(

        df.index.values[0], periods=num_idx, freq='T'))

    combined_df = pd.concat([df, idx_df], axis=1, sort=False)



    result_df = combined_df.fillna(method='bfill').resample('T').mean()

    if len(result_df) > num_idx:

        result_df = result_df[:num_idx]



    result_df = result_df.resample('1T').mean()



    return result_df





def load_daily_csv(file, file_dir=EXPORT_DIR):

    """

    載入每日的 CSV 檔案，並且會補齊每分鐘的資料點

    :param file:

    :param file_dir:

    :return:

    """

    f = os.path.join(file_dir, file)

    df = read_csv(f, parse_dates=True, index_col=[0])



    return resample_data(df)





def load_cpu_data(start_dt, load_files=4, folder='', file_dir=EXPORT_DIR):

    """

    載入

    :param col:

    :param start_dt:

    :param load_files:

    :param folder:

    :return:

    """

    date = np.array(start_dt, dtype=np.datetime64)

    date_arr = date + np.arange(load_files)



    format_ = '%Y%m%d.csv'

    if folder != '':

        format_ = '{}/%Y%m%d.csv'.format(folder)

    csv_files = [pd.to_datetime(d).strftime(format_) for d in date_arr]



    all_df = [load_daily_csv(f, file_dir=file_dir) for f in csv_files]



    series = []

    for df in all_df:

        df = df.dropna()

        

        cpu = df['wmi_cpu_percent'].values * 100

        idx = df.index.values



        for i in range(cpu.shape[0]):

            dt = pd.to_datetime(idx[i])

            series.append([cpu[i]])



    series = np.array(series)

    #     print(df.head())

    #     series.append(df[col])



    # series = [x[col] for x in all_df]

    # print(series)



    # return np.concatenate(series)

    return series



def fit_model(model: Sequential, name, data, validate_data, epochs=50, steps_per_epoch=1, enableCallbacks=True):

#     logDir = "logs/scalars/" + ('%s -' % name) + \

#         datetime.now().strftime("%Y%m%d-%H%M%S")

#     tensorBoard_callback = TensorBoard(

#         log_dir=logDir, write_graph=False, write_images=False)

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=.00001, patience=10)



    callbacks = [earlyStopping]

    if enableCallbacks:

        callbacks.append(tensorBoard_callback)



    return model.fit_generator(data,

                               validation_data=validate_data,

                               steps_per_epoch=steps_per_epoch,

                               epochs=epochs,

                               verbose=1,

                               callbacks=[earlyStopping])
def scalar_data(s, data: np.ndarray) -> np.ndarray:

    """

    用 scalar 縮放資料

    :param s:

    :param data:

    :return:

    """

    shaped_data = data.reshape(len(data), 1)

    scaled_data = s.transform(shaped_data)



    return scaled_data.reshape(len(data))
# 載入訓練資料

train = load_cpu_data('2019-08-12', 10)

print('train shape: ', train.shape)



# 先設定好縮放比例

scalar = sklearn.preprocessing.MinMaxScaler()

scalar.fit(train.reshape(len(train), 1))



train = scalar.transform(train)



# print(train)

# print('train shape: ', train.shape)
# 載入驗證資料

test = load_cpu_data('2019-08-30', 1)

print('test shape:  ', test.shape)



test_cpu = np.array([x[-1] for x in test])

test = scalar.transform(test)
# 設定好後面的參數值

n_sample = 1

n_time_step = 120

n_batch_size = 1

n_features = 1

n_output = 1





def generator(data, target=None): 

    return get_generator(data, n_output, n_time_step, n_batch_size, target)
train_generator = generator(train, train)

test_generator = generator(test, test)



# (x, y) = train_generator[0]

# print(x, y)
# 設定好 Model

model = Sequential([

    CuDNNGRU(50, return_sequences=True, input_shape=(n_time_step, n_features)),

    Activation('relu'),

    Dropout(0.1),

    CuDNNGRU(25, return_sequences=True),

    Activation('relu'),

    Dropout(0.1),

    CuDNNGRU(10),

    Activation('relu'),

    Dropout(0.1),

#     Dense(10),

    Dense(n_output)

])



model.compile(optimizer='adam', loss='mse', metrics=['mse'])

print(model.summary())
# 訓練數量

n_epochs = 100



logName = ("[TimeStep:%s]" % n_time_step) + ("[F_%s]" % n_features) + "-GRU_200_100_10_FC_10"

history = fit_model(model,

                    logName,

                    train_generator,

                    test_generator,

                    epochs=n_epochs,

                    enableCallbacks=False)



print(history.history['loss'][-1])
plot.figure(figsize=(20,10))

plot.plot(history.history['loss'], label='val_loss')

plot.plot(history.history['val_loss'], 'ro', label='val_loss')
pred = load_cpu_data('2019-09-01', 1)

print(pred.shape)

# scaled_pred = scalar_data(scalar, pred)

# real_cpu = np.array([x[-1] for x in pred])

# print('real_cpu: ', real_cpu)



pred_ = scalar.transform(pred)



pred_generator = get_generator(pred_, n_output, n_time_step, n_batch_size)



yhat = []

for i in range(len(pred_generator)):

    pred_x, _ = pred_generator[i]

#     print(pred_x)

    y_ = model.predict(pred_x) 

    yhat.append(np.concatenate(y_)) 



# # 將預測的結果，轉換回陣列，要跟原始資料做比較圖

yhat_ = scalar.inverse_transform(np.array(yhat))
plot.figure(figsize=(20,10))

plot.plot(pred)

plot.plot(yhat_)