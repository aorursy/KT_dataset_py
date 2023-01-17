import pandas as pd

import numpy as np

import random

import os

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation

from tensorflow.keras.layers import LSTM,GRU,Bidirectional

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)
df = pd.read_excel('../input/4th-datarobot-ai-academy-deep-learning/sales_prediction.xlsx')



df['Sales_lag7'] = df['Sales'].shift(7)

df['Num_Customers_lag7'] = df['Num_Customers'].shift(7)

df['Num_Employees_lag7'] = df['Num_Employees'].shift(7)

df['Pct_On_Sale_lag7'] = df['Pct_On_Sale'].shift(7)

df['Pct_Promotional_lag7'] = df['Pct_Promotional'].shift(7)

df['Returns_Pct_lag7'] = df['Returns_Pct'].shift(7)



df = df[df['Sales_lag7'].notnull()]

df = df[df['Sales']>0].reset_index(drop=True)



display(df.head())

display(df.shape)
ts_cols = ['Sales_lag7', 'Num_Customers_lag7','Num_Employees_lag7', 'Pct_On_Sale_lag7', 

           'Pct_Promotional_lag7','Returns_Pct_lag7'] 

step = 7



# 欠損値補填

df[ts_cols] = df[ts_cols].fillna(0)



# 正規化

scaler = StandardScaler()

df[ts_cols] = scaler.fit_transform(df[ts_cols]) 



# 特徴量配列

ts_x = np.array([])

for i in range(df.shape[0]):

    if i + step == df.shape[0]:

        break

    ts_x = np.append(ts_x, df.loc[i: i+step-1, ts_cols])

    

print ('ts_x',ts_x.shape)
target = 'Sales'



# trainは最終一週間前のデータ、testは最終一週間のデータ

train = df[(df['Date']<='2014-06-07')].reset_index(drop=True)

test = df[(df['Date']>'2014-06-07')].reset_index(drop=True) 



# targetの開始位置は1 stepずらしてとる

train_y = train[step:][target].values

test_y = test[target].values



len_train = len(train)

len_test = len(test)

len_ts_cols = len(ts_cols)



# 特徴量の終了位置も1　stepずらしてとる

train_x_ts = ts_x[:(len_train-step)*step*len_ts_cols]

test_x_ts = ts_x[(len_train-step)*step*len_ts_cols:]



# 入力データを3次元に変換(サンプル数、タイムステップ、特徴量数)

train_x_ts = train_x_ts.reshape(len_train-step, step, len_ts_cols)

test_x_ts = test_x_ts.reshape(len_test, step, len_ts_cols)



print ('train_x_ts',train_x_ts.shape)

print ('test_x_ts',test_x_ts.shape)

print ('train_y',train_y.shape)

print ('test_y',test_y.shape)
def rnn(step, ts_cols):    

    model = Sequential()

    """

    演習:BiLSTM、GRUに変更してみてください

    from tensorflow.keras.layers import Bidirectional,GRU

    LSTM -> Bidirectional(LSTM)

    LSTM -> GRU

    """

    model.add(GRU(units=512, activation='relu', kernel_initializer='he_normal', 

                   return_sequences=True, input_shape=(step, len(ts_cols))))

    model.add(Dropout(0.1)) 

    model.add(GRU(units=256, activation='relu', kernel_initializer='he_normal', return_sequences=True))

    model.add(Dropout(0.1)) 

    model.add(GRU(units=128, activation='relu', kernel_initializer='he_normal', return_sequences=False))

    model.add(Dropout(0.1)) 

    model.add(Dense(256, activation='relu', kernel_initializer='he_normal',))

    model.add(Dropout(0.1)) 

    model.add(Dense(128, activation='relu', kernel_initializer='he_normal',))  

    model.add(Dropout(0.1)) 

    model.add(Dense(32, activation='relu', kernel_initializer='he_normal',))  

    model.add(Dropout(0.1)) 

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
# callback parameter

filepath = "rnn_best_model.hdf5" 

es = EarlyStopping(patience=5, mode='min', verbose=1) 

checkpoint = ModelCheckpoint(monitor='val_loss',filepath=filepath, save_best_only=True,mode='auto') 

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1,mode='min')



# 訓練実行

model = rnn(step,ts_cols)

history = model.fit(train_x_ts, train_y, batch_size=32, epochs=30, validation_data=(test_x_ts, test_y),

                 callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# load best model weights

model.load_weights(filepath)



# predict

test_pred = model.predict(test_x_ts, batch_size=32).reshape((-1,1))

pred_score = mean_absolute_percentage_error(test_y,  test_pred)

print ('test mape:',pred_score)
model.summary()
plot_model(model, to_file='rnn.png')
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo' ,label = 'training loss')

plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()