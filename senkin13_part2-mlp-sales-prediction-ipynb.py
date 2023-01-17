import pandas as pd

import numpy as np

import random

import os

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler  

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(2020)    
df = pd.read_excel('../input/4th-datarobot-ai-academy-deep-learning/sales_prediction.xlsx')

# Cleasing

df = df[df['Sales']>0].reset_index(drop=True)

df['Holiday'] = df['Holiday'].map(lambda x: 0 if x=='No' else 1)

df['DestinationEvent'] = df['DestinationEvent'].map(lambda x: 0 if x=='No' else 1)

# Calendar Feature

df['year'] = df['Date'].dt.year

df['quarter'] = df['Date'].dt.quarter

df['month'] = df['Date'].dt.month

df['weekofoyear'] = df['Date'].dt.weekofyear

df['dayoyear'] = df['Date'].dt.dayofyear

df['dayofweek'] = df['Date'].dt.dayofweek

df['weekend'] = (df['Date'].dt.weekday >=5).astype(int)

df['dayofmonth'] = df['Date'].dt.day

# Lag Feature

df['Sales_lag7'] = df['Sales'].shift(7)

df['Num_Customers_lag7'] = df['Num_Customers'].shift(7)

df['Num_Employees_lag7'] = df['Num_Employees'].shift(7)

df['Pct_On_Sale_lag7'] = df['Pct_On_Sale'].shift(7)

df['Pct_Promotional_lag7'] = df['Pct_Promotional'].shift(7)

df['Returns_Pct_lag7'] = df['Returns_Pct'].shift(7)



display(df.head())

display(df.columns.values)
# 特徴量

num_cols = ['chrismas', 'blackfriday',

       'Holiday', 'DestinationEvent','year', 'quarter', 'month',

       'weekofoyear', 'dayoyear', 'dayofweek', 'weekend', 'dayofmonth',

       'Sales_lag7', 'Num_Customers_lag7', 'Num_Employees_lag7',

       'Pct_On_Sale_lag7', 'Pct_Promotional_lag7', 'Returns_Pct_lag7']

target = ['Sales']



# 欠損値補填

df[num_cols] = df[num_cols].fillna(0)



# 正規化

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])
# 訓練　検定　テストデータを分割

train = df[df['Date']<='2014-06-07']

valid = df[df['Date']>'2014-06-07'] 



# 特徴量とターゲット

train_x_num,train_y = train[num_cols].values,train[target].values

valid_x_num,valid_y = valid[num_cols].values,valid[target].values



print (train_x_num.shape)

print (valid_x_num.shape)
def mlp(num_cols):

    """

    演習:Dropoutを変更してみてください

    """

    model = Sequential()

    model.add(Dense(units=512, input_shape = (len(num_cols),), 

                    kernel_initializer='he_normal',activation='relu'))    

    model.add(Dropout(0.2))

    model.add(Dense(units=256,  kernel_initializer='he_normal',activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))     

    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    return model
filepath = "mlp_best_model.hdf5" 



"""

演習:patienceを変更してみてください

"""

es = EarlyStopping(patience=2, mode='min', verbose=1) 



checkpoint = ModelCheckpoint(monitor='val_loss',filepath=filepath, save_best_only=True, mode='auto') 



reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, verbose=1, mode='min')



model = mlp(num_cols)



"""

演習:batch_size,epochsを変更してみてください

"""

history = model.fit(train_x_num, train_y, batch_size=32, epochs=100, validation_data=(valid_x_num, valid_y), 

                    callbacks=[es, checkpoint, reduce_lr_loss], verbose=1)
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# load best model weights

model.load_weights(filepath)



# predict valid data

valid_pred = model.predict(valid_x_num, batch_size=32).reshape((-1,1))

valid_score = mean_absolute_percentage_error(valid_y,  valid_pred)

print ('valid mape:',valid_score)

model.summary()
plot_model(model, to_file='mlp.png')
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo' ,label = 'training loss')

plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()