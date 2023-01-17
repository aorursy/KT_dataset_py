import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context='notebook',

        style='whitegrid',

        palette='deep',

        font='sans-serif',

        font_scale=1,

        color_codes=True,

        rc=None)





from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf

from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.graphics.tsaplots as sgt

import statsmodels.tsa.stattools as sts
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM



import datetime, os

from keras.preprocessing.sequence import TimeseriesGenerator
tf.__version__
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/web-traffic-data-set/train_1.csv')

train = train_data

train.head(3)
train.info()
%%time

train = train.fillna(method='ffill', downcast='infer')

train.tail(3)
%%time

for cols in train.columns[1:]:

    train[cols] = pd.to_numeric(train[cols], downcast='integer')
train.info()
df = pd.DataFrame(train.iloc[:,1:].values.T,

            columns=train.Page.values, index=train.columns[1:])

df.index = pd.to_datetime(df.index, errors='ignore',

                                            dayfirst=False,

                                            yearfirst=False,

                                            utc=None,

                                            format="%Y/%m/%d",

                                            exact=False,

                                            unit=None,

                                            infer_datetime_format=True,

                                            origin='unix',

                                            cache=True)

df.head(3)
list(df.columns)[:10]  # First 10 pages
wikipedia = (df.filter(like='wikipedia'))

wikipedia
wikipedia.iloc[:,0:10].plot(figsize=(20,10))

plt.show()
def get_language(page):

    res = re.search('[a-z][a-z].wikipedia.org',page)

    if res:

        return res[0][0:2]

    return 'other'



(wikipedia.columns.map(get_language)).unique()
len((wikipedia.columns.map(get_language)).unique())
languages = list((wikipedia.columns.map(get_language)).unique())

languages.remove('other')

languages
for lang in (languages):

    locals()['lang_'+str(lang)] = wikipedia.loc[:, wikipedia.columns.str.contains('_'+str(lang)+'.wiki')]
lang_en.head(3)
for lang in (languages):

    locals()['hits_'+str(lang)] = np.array(locals()['lang_'+str(lang)].iloc[:,:].sum(axis=1))
for lang in (languages):

    print((locals()['hits_'+str(lang)]).shape)
keys = languages

values = ['Chinese', 'French', 'English', 'Russian', 'German', 'Japanese', 'Spanish']
d = dict(zip(keys,values))
index = wikipedia.index



hits = pd.DataFrame(index=index, columns=list(d.values()))

hits = hits.fillna(0)
for key, value in d.items():

    hits[value] = locals()['hits_'+str(key)]
hits
hits.plot(figsize=(25,8), title ='Hits on Wikipedia pages per Language', fontsize=15)

plt.legend(loc='upper left')

plt.show()
plt.rcParams["figure.dpi"] = 100

hits.iloc[:,0:1].plot(figsize=(20,4))

sgt.plot_acf(np.array(hits.iloc[:,0:1]),

            ax=None,

            lags=None,

            alpha=0.05,

            use_vlines=True,

            unbiased=False,

            fft=False,

            missing='none',

            title='Autocorrelation',

            zero=False,  # Not including the 1st term as its acf w.r.t. itself will always be 1.

            vlines_kwargs=None)

plt.show()
plt.rcParams["figure.dpi"] = 100

hits.iloc[:,0:1].plot(figsize=(20,4))

sgt.plot_pacf(np.array(hits.iloc[:,0:1]),

            ax=None,

            lags=None,

            alpha=0.05,

            method='ols',

            use_vlines=True,

            title='Partial Autocorrelation',

            zero=False,    # Not including the 1st term as its pacf w.r.t. itself will always be 1.

            vlines_kwargs=None)

plt.show()
brk = 0.8

data_split = int(len(hits)*brk)

data_split
X, y = hits.iloc[:data_split,:], hits.iloc[data_split:,:]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)



scaled_X = scaler.transform(X)

scaled_y = scaler.transform(y)
print(scaled_X.max(), scaled_X.min())

print(scaled_y.max(), scaled_y.min())
X_df = (pd.DataFrame(scaled_X))

y_df = (pd.DataFrame(scaled_y))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,8), dpi=100)

plt.suptitle('Train-Test Split', fontsize=20)

X_df.plot(ax=axes[0], title='Train Data')

y_df.plot(ax=axes[1], title='Test Data')



plt.show()
pd.DataFrame(scaled_y[3:13,:]).plot(figsize=(15,5), title='Periodicity')

plt.show()
print(scaled_X.shape)

print(scaled_y.shape)

print('No. of features = '+str(scaled_X.shape[1]))

print('No. of train instances = '+str(scaled_X.shape[0]))

print('No. of test instances = '+str(scaled_y.shape[0]))
length = 7

batch = 1



n_features = scaled_X.shape[1]

n_features
generator = TimeseriesGenerator(data = scaled_X,

                                targets = scaled_X,

                                length = length,

                                sampling_rate=1,

                                stride=1,

                                start_index=0,

                                end_index=None,

                                shuffle=False,

                                reverse=False,

                                batch_size=batch)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM



import datetime, os



model = Sequential(layers=None, name="LSTM_Model")



model.add(LSTM( units = 400,               

                activation='tanh',

                input_shape=( length, n_features),                

                recurrent_activation='sigmoid',

                use_bias=True,

                kernel_initializer='glorot_uniform',

                recurrent_initializer='orthogonal',

                bias_initializer='zeros',

                unit_forget_bias=True,

                kernel_regularizer=None,

                recurrent_regularizer=None,

                bias_regularizer=None,

                activity_regularizer=None,

                kernel_constraint=None,

                recurrent_constraint=None,

                bias_constraint=None,

                dropout=0.0,

                recurrent_dropout=0.0,

                implementation=2,

                return_sequences=True,

                return_state=False,

                go_backwards=False,

                stateful=False,

                time_major=False,

                unroll=False

            ) )

model.add(LSTM(units = 500, return_sequences=True))



model.add(LSTM(units = 500, return_sequences=False))



model.add(Dense(700, activation="relu", name="layer1"))



model.add(Dense(100, activation="relu", name="layer2"))





model.add(Dense( units = n_features,               

                activation='relu',

                use_bias=True,                        

                kernel_initializer='glorot_uniform',  

                bias_initializer='zeros',             

                kernel_regularizer=None,              

                bias_regularizer=None,                

                activity_regularizer=None,            

                kernel_constraint=None,               

                bias_constraint=None))                







model.compile(optimizer='adam', loss='mse')
model.summary()
from tensorflow.keras.callbacks import EarlyStopping



early_stop = EarlyStopping(monitor='val_loss',

                        min_delta=0,

                        patience=20,

                        verbose=1,  

                        mode='auto',

                        baseline=None,  

                                               

                        restore_best_weights=False)
validation_generator = TimeseriesGenerator(scaled_y,scaled_y, length=length, batch_size=batch)
%%time

history = model.fit(generator,

                    steps_per_epoch=None,

                    epochs=500,

                    verbose=1,

                    callbacks=[early_stop],

                    validation_data = validation_generator,

                    validation_steps=None,

                    validation_freq=1,

                    class_weight=None,

                    max_queue_size=10,

                    workers=1,

                    use_multiprocessing=False,

                    shuffle=True,

                    initial_epoch=0)
print(history.history.keys())
losses = pd.DataFrame(model.history.history)
plt.rcParams["figure.dpi"] = 100

losses.plot(figsize=(10,5))

plt.title('Losses')

plt.show()
%%time

test_predictions = []



first_eval_batch = scaled_X[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(scaled_y)):

    

    current_pred = model.predict(current_batch,verbose=0)[0]

    

    test_predictions.append(current_pred) 

    

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
print(np.array(test_predictions).shape)

print(scaled_y.shape)
print(np.array(test_predictions).max(), np.array(test_predictions).min())

print(scaled_y.max(), scaled_y.min())
true_predictions = scaler.inverse_transform(test_predictions)

print(true_predictions.shape)
print(np.array(true_predictions).max(), np.array(true_predictions).min())

print(np.array(y).max(), np.array(y).min())
t_l = len(scaled_y)

t_l
plt.figure(dpi=100)

plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,0:1] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,0:1], label='Predicted Values',c='r')

plt.title(hits.columns[0], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,1:2] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,1:2], label='Predicted Values',c='r')

plt.title(hits.columns[1], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,2:3] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,2:3], label='Predicted Values',c='r')

plt.title(hits.columns[2], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,3:4] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,3:4], label='Predicted Values',c='r')

plt.title(hits.columns[3], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,4:5] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,4:5], label='Predicted Values',c='r')

plt.title(hits.columns[4], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,5:6] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,5:6], label='Predicted Values',c='r')

plt.title(hits.columns[5], fontsize=20)

plt.legend()

plt.show()
plt.plot(np.linspace(0,t_l,t_l), scaled_y[:,6:7] , label='True Values',c='g')

plt.plot(np.linspace(0,t_l,t_l), np.array(test_predictions)[:,6:7], label='Predicted Values',c='r')

plt.title(hits.columns[6], fontsize=20)

plt.legend()

plt.show()