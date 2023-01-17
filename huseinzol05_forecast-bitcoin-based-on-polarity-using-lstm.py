import pandas as pd
import re 
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import os
print(os.listdir("../input"))

notclean = pd.read_csv('../input/bitcoin-tweets-14m/cleanprep.csv', delimiter=',', error_bad_lines=False,engine = 'python',header = None)
notclean.columns =['dt', 'name','text','polarity','sensitivity']
notclean =notclean.drop(['name','text'], axis=1)
notclean['dt'] = pd.to_datetime(notclean['dt'])
notclean['DateTime'] = notclean['dt'].dt.floor('h')
vdf = notclean.groupby(pd.Grouper(key='dt',freq='H')).size().reset_index(name='tweet_vol')
vdf.index = pd.to_datetime(vdf.index)
vdf=vdf.set_index('dt')
notclean.index = pd.to_datetime(notclean.index)
vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)
df = notclean.groupby('DateTime').agg(lambda x: x.mean())
df = df.drop(df.index[0])
btcDF = pd.read_csv('../input/btc-price/btcSave2.csv', error_bad_lines=False,engine = 'python')
btcDF['Timestamp'] = pd.to_datetime(btcDF['Timestamp'])
btcDF = btcDF.set_index(pd.DatetimeIndex(btcDF['Timestamp']))
btcDF = btcDF.drop(['Timestamp'], axis=1)
Final_df = pd.merge(df,btcDF, how='inner',left_index=True, right_index=True)
Final_df=Final_df.drop(['Weighted Price'],axis=1 )
print(list(Final_df))
Final_df.columns = ['Polarity', 'Sensitivity', 'Open','High','Low', 'Close_Price', 'Volume_BTC', 'Volume_Dollar']
Final_df = Final_df[['Polarity', 'Sensitivity', 'Open','High','Low', 'Volume_BTC', 'Volume_Dollar', 'Close_Price']]
Final_df.head()
df = Final_df
df['Polarity'].describe()
df['Sensitivity'].describe()
df['Close_Price'].describe()
def detect(signal,treshold=2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected
signal = np.copy(df['Close_Price'].values)
std_signal = (signal - np.mean(signal)) / np.std(signal)
s = pd.Series(std_signal)
s.describe(percentiles=[0.25,0.5,0.75,0.95])
outliers = detect(std_signal, 1.3)
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure(figsize=(15,7))
plt.plot(np.arange(len(signal)), signal)
plt.plot(np.arange(len(signal)), signal, 'X', label='outliers',markevery=outliers, c='r')
plt.xticks(np.arange(len(signal))[::15], df.index[::15], rotation='vertical')
plt.show()
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(df[['Polarity','Sensitivity','Close_Price']])
scaled = minmax.transform(df[['Polarity','Sensitivity','Close_Price']])
plt.figure(figsize=(15,7))
plt.plot(np.arange(len(signal)), scaled[:,0], label = 'Scaled polarity')
plt.plot(np.arange(len(signal)), scaled[:,1], label = 'Scaled sensitivity')
plt.plot(np.arange(len(signal)), scaled[:,2], label = 'Scaled closed price')
plt.plot(np.arange(len(signal)), scaled[:,0], 'X', label='outliers polarity based on close', markevery=outliers, c='r')
plt.plot(np.arange(len(signal)), scaled[:,1], 'o', label='outliers polarity based on close', markevery=outliers, c='r')
plt.xticks(np.arange(len(signal))[::15], df.index[::15], rotation='vertical')
plt.legend()
plt.show()
colormap = plt.cm.RdBu
plt.figure(figsize=(15,7))
plt.title('pearson correlation', y=1.05, size=16)

mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df.corr(), mask=mask, linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
def df_shift(df,lag=0, start=1, skip=1, rejected_columns = []):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(start,lag+1,skip):
        for x in list(df.columns):
            if x not in rejected_columns:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
    return df
df_new = df_shift(df, lag=42, start=7, skip=7)
df_new.shape
colormap = plt.cm.RdBu
plt.figure(figsize=(30,20))
ax=plt.subplot(111)
plt.title('42 hours correlation', y=1.05, size=16)
selected_column = [col for col in list(df_new) if any([k in col for k in ['Polarity','Sensitivity','Close']])]

sns.heatmap(df_new[selected_column].corr(), ax=ax, linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
def moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer
signal = np.copy(df['Close_Price'].values)
ma_7 = moving_average(signal, 7)
ma_14 = moving_average(signal, 14)
ma_30 = moving_average(signal, 30)
plt.figure(figsize=(15,7))
plt.plot(np.arange(len(signal)), signal, label='real signal')
plt.plot(np.arange(len(signal)), ma_7, label='ma 7')
plt.plot(np.arange(len(signal)), ma_14, label='ma 14')
plt.plot(np.arange(len(signal)), ma_30, label='ma 30')
plt.legend()
plt.show()
num_layers = 1
learning_rate = 0.005
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.6
dates = pd.to_datetime(df.index).tolist()
class Model:
    def __init__(self, learning_rate, num_layers, 
                 size, size_layer, forget_bias = 0.8):
        
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)], 
                                                state_is_tuple = False)
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, size))
        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob = forget_bias)
        self.hidden_layer = tf.placeholder(tf.float32, 
                                           (None, num_layers * 2 * size_layer))
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X, 
                                                          initial_state = self.hidden_layer, 
                                                          dtype = tf.float32)
        self.logits = tf.layers.dense(self.outputs[-1],size,
                       kernel_initializer=tf.glorot_uniform_initializer())
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
minmax = MinMaxScaler().fit(df[['Polarity','Sensitivity','Close_Price']].astype('float32'))
df_scaled = minmax.transform(df[['Polarity','Sensitivity','Close_Price']].astype('float32'))
df_scaled = pd.DataFrame(df_scaled)
df_scaled.head()
tf.reset_default_graph()
modelnn = Model(learning_rate, num_layers, df_scaled.shape[1], size_layer, dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(epoch):
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, (df_scaled.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(df_scaled.iloc[k: k + timestamp].values, axis = 0)
        batch_y = df_scaled.iloc[k + 1: k + timestamp + 1].values
        last_state, _, loss = sess.run([modelnn.last_state, 
                                        modelnn.optimizer, 
                                        modelnn.cost], feed_dict={modelnn.X: batch_x, 
                                                                  modelnn.Y: batch_y, 
                                                                  modelnn.hidden_layer: init_value})
        init_value = last_state
        total_loss += loss
    total_loss /= (df.shape[0] // timestamp)
    if (i + 1) % 100 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)
def predict_future(future_count, df, dates, indices={}):
    date_ori = dates[:]
    cp_df = df.copy()
    output_predict = np.zeros((cp_df.shape[0] + future_count, cp_df.shape[1]))
    output_predict[0, :] = cp_df.iloc[0]
    upper_b = (cp_df.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    for k in range(0, (df.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    cp_df.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(cp_df.iloc[upper_b:], axis = 0),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[upper_b + 1 : cp_df.shape[0] + 1] = out_logits
    cp_df.loc[cp_df.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(hours = 1))
    if indices:
        for key, item in indices.items():
            cp_df.iloc[-1,key] = item
    for i in range(future_count - 1):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(cp_df.iloc[-timestamp:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[cp_df.shape[0], :] = out_logits[-1, :]
        cp_df.loc[cp_df.shape[0]] = out_logits[-1, :]
        date_ori.append(date_ori[-1] + timedelta(hours = 1))
        if indices:
            for key, item in indices.items():
                cp_df.iloc[-1,key] = item
    return {'date_ori': date_ori, 'df': cp_df.values}    
def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
predict_30 = predict_future(30, df_scaled, dates)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])
plt.figure(figsize=(15,7))
plt.plot(np.arange(len(predict_30['date_ori'])), anchor(predict_30['df'][:,-1],0.5), label='predict signal')
plt.plot(np.arange(len(signal)), signal, label='real signal')
plt.legend()
plt.show()
scaled_polarity = (minmax.data_max_[0] * 2 - minmax.data_min_[0]) / (minmax.data_max_[0] - minmax.data_min_[0])
scaled_polarity
plt.figure(figsize=(15,7))

for retry in range(3):
    plt.subplot(3, 1, retry + 1)
    predict_30 = predict_future(30, df_scaled, dates, indices = {0:scaled_polarity})
    predict_30['df'] = minmax.inverse_transform(predict_30['df'])
    plt.plot(np.arange(len(predict_30['date_ori'])), anchor(predict_30['df'][:,-1],0.5), label='predict signal')
    plt.plot(np.arange(len(signal)), signal, label='real signal')
    plt.legend()
plt.show()
scaled_polarity = (minmax.data_max_[0] * 4 - minmax.data_min_[0]) / (minmax.data_max_[0] - minmax.data_min_[0])
scaled_polarity
plt.figure(figsize=(15,7))

for retry in range(3):
    plt.subplot(3, 1, retry + 1)
    predict_30 = predict_future(30, df_scaled, dates, indices = {0:scaled_polarity})
    predict_30['df'] = minmax.inverse_transform(predict_30['df'])
    plt.plot(np.arange(len(predict_30['date_ori'])), anchor(predict_30['df'][:,-1],0.5), label='predict signal')
    plt.plot(np.arange(len(signal)), signal, label='real signal')
    plt.legend()
plt.show()
scaled_polarity = (minmax.data_min_[0] / 4 - minmax.data_min_[0]) / (minmax.data_max_[0] - minmax.data_min_[0])
scaled_polarity
plt.figure(figsize=(15,7))

for retry in range(3):
    plt.subplot(3, 1, retry + 1)
    predict_30 = predict_future(30, df_scaled, dates, indices = {0:scaled_polarity})
    predict_30['df'] = minmax.inverse_transform(predict_30['df'])
    plt.plot(np.arange(len(predict_30['date_ori'])), anchor(predict_30['df'][:,-1],0.5), label='predict signal')
    plt.plot(np.arange(len(signal)), signal, label='real signal')
    plt.legend()
plt.show()
