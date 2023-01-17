import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/work2210/1.xlsx')
df
df.drop(['Номер', 'Исходный профиль', 'Профиль'], axis=1, inplace=True)
df
df['Количество пикселей на входе'] = df['Входная высота, пиксели'] * df['Входная ширина, пиксели']
df['Количество пикселей на выходе'] = df['Выходная высота, пиксели'] * df['Выходная ширина, пиксели']
df.drop(['Входная высота, пиксели', 'Входная ширина, пиксели', 'Выходная высота, пиксели', 'Выходная ширина, пиксели'], 
        axis=1, inplace=True)
df
df['Отношение пикселей'] = df['Количество пикселей на выходе'] / df['Количество пикселей на входе']
df.drop(['Количество пикселей на выходе', 'Количество пикселей на входе'], axis=1, inplace=True)
df
df['Размер, Мб'] = df['Размер, Мб'].astype(np.float32)
df['Таргет'] = df['Таргет'].astype(np.float32)

df
df = pd.get_dummies(data=df, columns=['Кодек', 'Битрейт'])
df
df.info()
df['Скорость битрейта'].unique(), df['Качество битрейта'].unique()
df['Скорость битрейта'] = df.apply(lambda row: 0 if row['Скорость битрейта'] == '-' else row['Скорость битрейта'], axis=1)
df['Качество битрейта'] = df.apply(lambda row: 0 if row['Качество битрейта'] == '-' else row['Качество битрейта'], axis=1)
df
df['Скорость битрейта'].unique(), df['Качество битрейта'].unique()
df['Скорость битрейта'] = df['Скорость битрейта'].astype(np.int32)
df['Качество битрейта'] = df['Качество битрейта'].astype(np.int32)
df
df.drop(['Длина, секунды'], axis=1, inplace=True)
df
Y = df['Таргет'].values
df.drop(['Таргет'], axis=1, inplace=True)
df
df['Качество битрейта'] /= df['Качество битрейта'].max()
df['Скорость битрейта'] /= df['Скорость битрейта'].max()
df
df.info()
df.iloc[:, 4:] = df.iloc[:, 4:].astype(np.int32)
df.info()
X = df.values
# X = X.astype(np.float32)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train.shape
Y_train
import keras
from keras.models import Sequential, load_model 
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, RepeatVector, TimeDistributed, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import models, optimizers, Model
import tensorflow as tf
import tensorflow_addons as tfa


model = Sequential()

model.add(Dense(20, kernel_initializer='normal', activation='relu', input_shape=(X_train.shape[1],) ))
model.add(Dropout(0.3))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(80, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.8))

# model.add(Dense(40, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.6))

# model.add(Dense(40, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))
model.compile(optimizer=optimizers.Adam(lr=1e-2), loss='mse', metrics=['mape'])

model.summary()
from keras.callbacks import ModelCheckpoint

def lr_scheduler(epoch, lr):
    return lr * 0.95

checkpoint_path = 'bestmodel.hdf5'

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mse', verbose=0, save_best_only=True, mode='min')

scheduler = LearningRateScheduler(lr_scheduler, verbose=0)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, mode='min', verbose=1)

tqdm_callback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False, 
                                              leave_overall_progress=True, 
                                              show_epoch_progress=False,
                                              show_overall_progress=True)

callbacks_list = [
    checkpoint, 
    scheduler, 
    early_stop, 
#    tqdm_callback
]

history = model.fit(X_train, Y_train, epochs=100, batch_size=8, callbacks=callbacks_list, verbose=1, validation_split=0.2)
import matplotlib.pyplot as plt

def graph_plot(history):
    fig = plt.figure(figsize=(16, 24))
    for i in history.history.keys():
        print(f'{i} = [{min(history.history[i])}; {max(history.history[i])}]\n')
    
    epoch = len(history.history['loss'])
    # на каждую: (train, val) + lr
    size = len(history.history.keys()) // 2 + 1
    
    i = 1
    for k in list(history.history.keys()):
        if 'val' not in k:
            fig.add_subplot(size, 1, i)
            plt.plot(history.history[k])
            if k != 'lr':
                plt.plot(history.history['val_' + k])
            plt.title(k, fontsize=10)

            plt.ylabel(k)
            plt.xlabel('epoch')
            plt.grid()

            plt.yticks(fontsize=10, rotation=30)
            plt.xticks(fontsize=10, rotation=30)
            plt.legend(['train', 'valid'], loc='upper left', fontsize=10, title_fontsize=15)
            i += 1
#         plt.show()

graph_plot(history)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, Y_train)
reg.score(X_train, Y_train)
reg.predict(X_test)
Y_test
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'n_estimators': range(3, 41, 1),
        'max_features': range(3, 13, 1),
        'bootstrap': [True, False]

    }
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
grid_search.best_params_ 
grid_search.best_estimator_ 
best = grid_search.best_estimator_ 
pred = best.predict(X_test)
pred
Y_test
from sklearn.metrics import mean_absolute_percentage_error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(Y_test, pred)
df
X_test2 = np.array([[116, 30 / 31, 0.0, 1.0, 1, 0, 0, 0, 0, 1, 0, 0]])
best.predict(X_test2)
from xgboost import XGBRegressor


params = {
    'min_child_weight': range(2, 6, 1), 
    'gamma': list(np.logspace(-4, -1, 4)),
    'subsample': [i / 10.0 for i in range(6, 11)],
    'colsample_bytree': [i / 10.0 for i in range(6, 21)], 
    'max_depth': range(2, 9, 1),
    'learning_rate': np.logspace(-4, -1, 4),
    'n_estimators': [10, 100]
}


xgb = XGBRegressor(nthread=-1) 

grid = GridSearchCV(xgb, params, cv=4, n_jobs=-1, scoring='neg_mean_squared_error')
grid.fit(X_train, Y_train)
pred = grid.best_estimator_.predict(X_test)
mean_absolute_percentage_error(Y_test, pred)
grid.best_estimator_.predict(X_test2)
grid.best_params_
