import numpy as np
import pandas as pd
from datetime import datetime 
init_time = datetime.now()
# CONSTANTS

IS_DEV = True
TREAT_OUTLIERS = False

if(IS_DEV == True):
    TRAIN_FILE = '../input/train.csv'
    TEST_FILE = '../input/test.csv'
    COMPRESSION = None
    SKIP_ROWS = range(1, 170000000)
else:
    TRAIN_FILE = '../input/train.csv.zip'
    TEST_FILE = '../input/test.csv.zip'
    COMPRESSION = 'zip'
    
TRAIN_SAMPLE_FILE = '../input/' + 'train_sample.csv'
    
LEARNING_RATE_INIT = 0.001
LEARNING_RATE_END = 0.0001
BATCH_SIZE = 20000
EPOCHS = 2

EMBEDDING_N = 50
DENSE_N = 1024

SPATIAL_DROPOUT_1D = 0.2
DROPOUT_1 = 0.2
DROPOUT_2 = 0.2
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
# Ensure file path is correct.
train_sample = pd.read_csv(    
    TRAIN_SAMPLE_FILE, 
    dtype=dtypes,
    usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
    compression = None,
    header=0,
    #skiprows=range(1, 181886954),  # 2016-01-01)
    engine='c'
)
train_sample.head()
# Load Data
print('Loading data...')
start_time = datetime.now()
train = pd.read_csv(    
    TRAIN_FILE, 
    dtype=dtypes,
    usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
    compression = COMPRESSION,
    header=0,
    skiprows=SKIP_ROWS,  # 2016-01-01)
    engine='c'
)

test = pd.read_csv(    
    TEST_FILE, 
    dtype=dtypes,
    usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'],
    compression = COMPRESSION,
    header=0,
    #skiprows=range(1, 17886954),  # 2016-01-01)
    engine='c'
)

print('End of Loading Data: {time_taken}'.format(time_taken=datetime.now() - start_time))
train.head()
if(False):

    import seaborn as sns
    import matplotlib.pyplot as plt

    from mlxtend.preprocessing import minmax_scaling
    from scipy import stats


    positive_is_attributed = train.is_attributed.loc[train.is_attributed > 0]

    # Normalize
    normalized_is_attributed = stats.boxcox(positive_is_attributed)[0]
    fig, ax = plt.subplots(1, 3)
    sns.distplot(train.is_attributed, ax=ax[0])
    ax[0].set_title('Original Data')
    sns.distplot(normalized_is_attributed[0], ax=ax[1])
    ax[1].set_title('Normalized Data')
# Remove Outliers
print(len(train))

class Outliers():

    data = None
    def __init__(self, data):
        """
        INPUT
        data - Dataframe having date column in MM-DD-YYYY format
        """
        self.data = data

    def treat_outliers_by_iqr(self, colName, what_sd=1.5):
        mean_value = np.mean(self.data[colName].values)
        sd_value = np.std(self.data[colName].values)

        max_value = mean_value + what_sd * sd_value
        min_value = mean_value - what_sd * sd_value

        new_data = self.data[(self.data[colName] >= min_value) & (self.data[colName] <= max_value)]
        outliers = self.data[(self.data[colName] > max_value) | (self.data[colName] < min_value)]

        print('%Outliers: {outlier_percentage:<8} Mean: {mean_value:<8} SD: {sd_value:<8} IQR Min: {iqr_min:<8} IQR Max: {iqr_max:<8} Min: {mini:<8} Max: {maxi:<8}  Outliers: {outliers:<8} #App: {app_count:<8} #Obs: {size}'.format(
            outlier_percentage=np.around((len(outliers)/len(self.data))*100, decimals=2), 
            mean_value=np.around(mean_value, decimals=2), sd_value=np.around(sd_value, decimals=2), 
            iqr_min=np.around(min_value, decimals=2),
            iqr_max=np.around(max_value, decimals=2),
            mini=np.around(min(self.data[colName].values), decimals=2), 
            maxi=np.around(max(self.data[colName].values), decimals=2),
            outliers=len(outliers),
            app_count=len(new_data.app.unique()), 
            size=len(new_data)
        ))

        return outliers, new_data
    
if(TREAT_OUTLIERS == True):   
    print('Remove Outliers...')
    start_time = datetime.now()

    outs = Outliers(train)
    outliers, train = outs.treat_outliers_by_iqr('is_attributed')

    print('End of Outlier Removal: {time_taken}'.format(time_taken=datetime.now() - start_time))
train_size = len(train)
train = train.append(test)
del test

import gc
gc.collect()
# Datetime Object
print('Making Hour, Day and Weekday Columns...')
start_time = datetime.now()

date_time = pd.to_datetime(train.click_time)

train['hour'] = date_time.dt.hour.astype('uint8')
train['day'] = date_time.dt.day.astype('uint8')
train['wday'] = date_time.dt.dayofweek.astype('uint8')

print('End of Making HDWd: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Making number of channels by IP, Day, HOUR...')
group_ip_dhc = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])
group_ip_dhc = group_ip_dhc[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train = train.merge(group_ip_dhc, on=['ip', 'day', 'hour'], how='left')

del group_ip_dhc
gc.collect

train.head()

print('End of Making channel count by IDH: {time_taken}'.format(time_taken=datetime.now() - start_time))
# Group by IP, APP, OS Combination
print('Making number of channels by IP, APP...')
start_time = datetime.now()

group_ip_ac = train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])
group_ip_ac = group_ip_ac[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train = train.merge(group_ip_ac, on=['ip', 'app'], how='left')

del group_ip_ac
gc.collect()

train.head()

print('End of Making channel count by IA: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Making number of channels by IP, APP, OS...')
start_time = datetime.now()

group_ip_aoc = train[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])
group_ip_aoc = group_ip_aoc[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train = train.merge(group_ip_aoc, on=['ip', 'app', 'os'], how='left')

del group_ip_aoc
gc.collect()

print('End of  Making channel count by IAO: {time_taken}'.format(time_taken=datetime.now() - start_time))
from sklearn.preprocessing import LabelEncoder
print('Label encoding and fit transformation...')

start_time = datetime.now()
train[['app', 'device', 'os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)

print('End of  Making channel count by IAO: {time_taken}'.format(time_taken=datetime.now() - start_time))
# train.head()
print('Splitting train and test data...')
start_time = datetime.now()

test = train[train_size:]
print(len(test))
train = train[:train_size]

y_train = train['is_attributed'].values
print(len(train))
train.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)

print('End of Splitting Train and Test Dataset: {time_taken}'.format(time_taken=datetime.now() - start_time))
# Neural Network
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GRU
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
max_app = np.max([train['app'].max(), test['app'].max()]) + 1
max_channel = np.max([train['channel'].max(), test['channel'].max()]) + 1
max_device = np.max([train['device'].max(), test['device'].max()]) + 1
max_os = np.max([train['os'].max(), test['os'].max()]) + 1
max_hour = np.max([train['hour'].max(), test['hour'].max()]) + 1
max_day = np.max([train['day'].max(), test['day'].max()]) + 1
max_wday = np.max([train['wday'].max(), test['wday'].max()]) + 1
max_qty = np.max([train['qty'].max(), test['qty'].max()]) + 1
max_c1 = np.max([train['ip_app_count'].max(), test['ip_app_count'].max()]) + 1
max_c2 = np.max([train['ip_app_os_count'].max(), test['ip_app_os_count'].max()]) + 1

print('Preparing dataset for training...')
start_time = datetime.now()

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'channel': np.array(dataset.channel),
        'device': np.array(dataset.device),
        'os': np.array(dataset.os),
        'hour': np.array(dataset.hour),
        'day': np.array(dataset.day),
        'wday': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X

train = get_keras_data(train)

in_app = Input(shape=[1], name='app')
emb_app = Embedding(max_app, EMBEDDING_N)(in_app)

in_channel = Input(shape=[1], name='channel')
emb_channel = Embedding(max_channel, EMBEDDING_N)(in_channel)

in_device = Input(shape=[1], name='device')
emb_device = Embedding(max_device, EMBEDDING_N)(in_device)

in_os = Input(shape=[1], name='os')
emb_os = Embedding(max_os, EMBEDDING_N)(in_os)

in_hour = Input(shape=[1], name='hour')
emb_hour = Embedding(max_hour, EMBEDDING_N)(in_hour)

in_day = Input(shape=[1], name='day')
emb_day = Embedding(max_day, EMBEDDING_N)(in_day)

in_wday = Input(shape=[1], name='wday')
emb_wday = Embedding(max_wday, EMBEDDING_N)(in_wday)

in_qty = Input(shape=[1], name='qty')
emb_qty = Embedding(max_qty, EMBEDDING_N)(in_qty)

in_c1 = Input(shape=[1], name='c1')
emb_c1 = Embedding(max_c1, EMBEDDING_N)(in_c1)

in_c2 = Input(shape=[1], name='c2')
emb_c2 = Embedding(max_c2, EMBEDDING_N)(in_c2)


print('End of Dataset Preparation: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Create RNN Layers...')
start_time = datetime.now()

rnn_layer_1 = GRU(16)(emb_app)
rnn_layer_2 = GRU(8)(emb_channel)
rnn_layer_3 = GRU(8)(emb_device)
rnn_layer_4 = GRU(8)(emb_os)

fe = concatenate([
    Flatten()(emb_app), 
    Flatten()(emb_channel), 
    Flatten()(emb_device), 
    Flatten()(emb_os), 
    Flatten()(emb_hour), 
    Flatten()(emb_day), 
    Flatten()(emb_wday), 
    Flatten()(emb_qty), 
    Flatten()(emb_c1), 
    Flatten()(emb_c2),
    rnn_layer_1,
    rnn_layer_2,
    rnn_layer_3,
    rnn_layer_4
])
print('End of RNN Layer Creation: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Building TF architecture...')
start_time = datetime.now()

#s_dout = SpatialDropout1D(SPATIAL_DROPOUT_1D)(fe)
#x = Flatten()(s_dout)
x = Dropout(DROPOUT_1)(Dense(DENSE_N, activation='relu')(fe))
x = Dropout(DROPOUT_1)(Dense(512, activation='relu')(x))
x = Dropout(DROPOUT_2)(Dense(256, activation='relu')(fe))
x = Dropout(DROPOUT_2)(Dense(128, activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app, in_channel, in_device, in_os,
    in_hour, in_day, in_wday, in_qty, in_c1, in_c2], outputs=outp)

print('End of Building TF Architecture: {time_taken}'.format(time_taken=datetime.now() - start_time))
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps -1)) - 1
steps = int(len(train) / BATCH_SIZE) * EPOCHS
lr_init, lr_fin = LEARNING_RATE_INIT, LEARNING_RATE_END
lr_decay = exp_decay(lr_init, lr_fin, steps)
print('Creating the model...')
start_time = datetime.now()

optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
model.summary()

print('End of Creating the Model: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Training...')
start_time = datetime.now()

class_weight = {0:0.01, 1:0.99}
model.fit(train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weight, shuffle=True, verbose=1)
del train, y_train
gc.collect()


print('End of Training: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Preparing test dataset...')
start_time = datetime.now()

submit = pd.DataFrame()
submit['click_id'] = test['click_id'].astype('int')
test.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)
test = get_keras_data(test)

print('End of Test Dataset: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('Preparing predicted output...')
start_time = datetime.now()

submit['is_attributed'] = model.predict(test, batch_size=BATCH_SIZE, verbose=2)
del test
gc.collect()
submit.to_csv('output.csv', index=False)

print('End of Predicted Output: {time_taken}'.format(time_taken=datetime.now() - start_time))
print('The END: {time_taken}'.format(time_taken=datetime.now() - init_time))
if(IS_DEV == False):
    import h5py
    model.save_weights('../output/weights.h5')
    model.save('../output/model.h5')
































