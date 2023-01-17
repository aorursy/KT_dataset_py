import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import pandas as pd
import seaborn as sns
import gc
train_col = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
df = pd.read_csv('../input/train.csv', nrows=30000000,
                 usecols=train_col, dtype=dtypes)
df.head()
df.describe()
df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df['dw'] = pd.to_datetime(df.click_time).dt.dayofweek.astype('uint8')

df_test = pd.read_csv('../input/test.csv')
click_id = df_test['click_id']
df_test['hour'] = pd.to_datetime(df_test.click_time).dt.hour.astype('uint8')
df_test['minute'] = pd.to_datetime(df_test.click_time).dt.minute.astype('uint8')
df_test['day'] = pd.to_datetime(df_test.click_time).dt.day.astype('uint8')
df_test['dw'] = pd.to_datetime(df_test.click_time).dt.dayofweek.astype('uint8')
from sklearn.preprocessing import LabelEncoder

X = pd.concat((df[['ip', 'app', 'device', 'hour', 'minute', 'os', 'channel', 'day', 'dw']],
               df_test[['ip', 'app', 'device', 'hour', 'minute', 'os', 'channel', 'day', 'dw']]))
del df_test; gc.collect()

group = X[['ip','day','hour', 'minute','channel']].groupby(by=['ip','day','hour', 'minute'])['channel'].count().\
reset_index().rename(index=str, columns={'channel': 'ip_day_time'})
X = X.merge(group, on=['ip','day','hour', 'minute'], how='left')
del group; gc.collect()

group = X[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])['channel'].count().\
reset_index().rename(index=str, columns={'channel': 'ip_app_os'})
X = X.merge(group, on=['ip', 'app', 'os'], how='left')
del group; gc.collect()


X[['app','device','os', 'channel', 'hour', 'minute', 'day', 'dw']].apply(LabelEncoder().fit_transform)

len(X)
max_app = np.max(X.app)+1
max_device = np.max(X.device)+1
max_hour = np.max(X.hour)+1
max_minute = np.max(X.minute)+1
max_os = np.max(X.os)+1
max_channel = np.max(X.channel)+1
max_day = np.max(X.day)+1
max_dw = np.max(X.dw)+1
max_ip_day_time = np.max(X.ip_day_time)+1
max_ip_app_os = np.max(X.ip_app_os)+1
from sklearn.model_selection import train_test_split
X_test = X[len(df):]
X = X[:len(df)]
y = df.is_attributed
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, train_size=0.95)
print(np.sum(y_train) / len(y_train))
print(np.sum(y_valid) / len(y_valid))
#from keras.preprocessing.sequence import pad_sequences

del df; gc.collect()
def get_keras_data(data):
    X = {
        'app': np.array(data.app),
        'device': np.array(data.device),
        'hour': np.array(data.hour),
        'minute': np.array(data.minute),
        'os': np.array(data.os),
        'channel': np.array(data.channel),
        'day': np.array(data.day),
        'dw': np.array(data.dw),
        'ip_day_time': np.array(data.ip_day_time),
        'ip_app_os': np.array(data.ip_app_os)
    }
    return X
X_train = get_keras_data(X_train)
X_valid = get_keras_data(X_valid)
X_test = get_keras_data(X_test)
from keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten, GRU
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.models import Model

def get_model():
    
    app = Input(shape=[1], name='app')
    device = Input(shape=[1], name='device')
    hour = Input(shape=[1], name='hour')
    minute = Input(shape=[1], name='minute')
    os = Input(shape=[1], name='os')
    channel = Input(shape=[1], name='channel')
    day = Input(shape=[1], name='day')
    dw = Input(shape=[1], name='dw')
    ip_day_time = Input(shape=[1], name='ip_day_time')
    ip_app_os = Input(shape=[1], name='ip_app_os')
        
    emb_app = Embedding(max_app, 30)(app)
    emb_device = Embedding(max_device, 30)(device)
    emb_hour = Embedding(max_hour, 30)(hour)
    emb_minute = Embedding(max_minute, 30)(minute)
    emb_os = Embedding(max_os, 30)(os)
    emb_channel = Embedding(max_channel, 30)(channel)
    emb_day = Embedding(max_day, 30)(day)
    emb_dw = Embedding(max_dw, 30)(dw)
    emb_ip_day_time = Embedding(max_ip_day_time, 30)(ip_day_time)
    emb_ip_app_os = Embedding(max_ip_app_os, 30)(ip_app_os)
    
    main_l = concatenate([Flatten()(emb_app), Flatten()(emb_hour), Flatten()(emb_minute),
                          Flatten()(emb_os), Flatten()(emb_device),
                          Flatten()(emb_day), Flatten()(emb_dw), Flatten()(emb_channel),
                          Flatten()(emb_ip_day_time), Flatten()(emb_ip_app_os)])
    main_l = Dropout(0.1)(Dense(1000, activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(1000, activation='relu')(main_l))
    
    output = Dense(1, activation='sigmoid')(main_l)
    
    model = Model([app, device, hour, minute, os, channel, day, dw, ip_day_time, ip_app_os], output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model

model = get_model()
model.summary()
    
batch_size = 20000
epochs = 1

model.fit(X_train, np.array(y_train), epochs=epochs, batch_size=batch_size, verbose=1)
pred = model.predict(X_valid)
from sklearn.metrics import auc, roc_auc_score, roc_curve

false_positive_rate, recall, thresholds = roc_curve(y_valid, pred)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()
del X; gc.collect()
pred_test = model.predict(X_test)
pred_test = pd.Series(pred_test.reshape(-1), name='is_attributed')
sub = pd.concat([click_id, pred_test], axis=1)
sub.to_csv('sub.csv', index=False)
sub.head()







