import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn

import numpy as np
import pandas as pd
pd.options.display.max_columns = 999

data_dir = '../input'
import os
print(os.listdir(data_dir))
# save space by using ints where possible
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

df = pd.read_csv('../input/train.csv', nrows=10000000, dtype=dtypes, parse_dates=['click_time'])
df = df.drop('attributed_time', axis=1)

df.head()
df['is_attributed'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('is_attributed', axis=1), 
                                                    df['is_attributed'], test_size=0.2)
def add_dateparts(df):
    print('Adding date parts...')
    df['day'] = df['click_time'].dt.dayofweek
    df['hour'] = df['click_time'].dt.hour
    df['minute'] = df['click_time'].dt.minute
    df['second'] = df['click_time'].dt.second
    return df
def add_ip_stats(df):
    GROUPBY_AGGREGATIONS = [
        # Click count within the minute of observation for same IP
        {'groupby': ['ip','day','hour', 'minute'], 'select': 'minute', 
         'agg': 'count', 'feature_name': 'clicks_count_same_minute'},

        # Click count within the hour of observation for same IP
        {'groupby': ['ip','day', 'hour'], 'select': 'hour', 
         'agg': 'count', 'feature_name': 'clicks_count_same_hour'},

        # Click count within the day of observation for same IP
        {'groupby': ['ip', 'day'], 'select': 'day', 
         'agg': 'count', 'feature_name': 'clicks_count_same_day'},
        
        # Number of device-os pairs used within the day of observation for same IP
        {'groupby': ['ip', 'day', 'device'], 'select': 'os', 
         'agg': 'count', 'feature_name': 'device_os_count_same_day'},
        
        # Number of app-channel pairs used within the day of observation for same IP
        {'groupby': ['ip', 'day', 'app'], 'select': 'channel', 
         'agg': 'count', 'feature_name': 'app_channel_count_same_day'},
        
        # Number of app-os pairs used within the day of observation for same IP
        {'groupby': ['ip', 'day', 'app'], 'select': 'os', 
         'agg': 'count', 'feature_name': 'app_os_count_same_day'}
    ]

    # Apply all the groupby transformations
    for spec in GROUPBY_AGGREGATIONS:
        print(f"Grouping by {spec['groupby']}, and aggregating {spec['select']} with {spec['agg']}")

        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # Perform the groupby
        result = df[all_features] \
                    .groupby(spec['groupby'])[spec['select']] \
                    .agg(spec['agg']) \
                    .rename(spec['feature_name']) \
                    .reset_index()

        # Merge back to X_train
        df = df.merge(result, on=spec['groupby'], how='left')
        
    return df
categorical_vars = ['app', 'device', 'os', 'channel', 'day', 'hour', 'minute', 'second']
numerical_vars = ['clicks_count_same_minute', 'clicks_count_same_hour', 'clicks_count_same_day', 
                  'device_os_count_same_day', 'app_channel_count_same_day','app_os_count_same_day']
drop_columns = ['ip', 'click_time']
target = 'is_attributed'
def feature_pipeline(X):
    X = add_dateparts(X)
    X = add_ip_stats(X)
    X = X.fillna(0)
    X = X.drop(drop_columns, axis=1)
    
    return X
feature_pipeline(X_train).head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = feature_pipeline(X_train)
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])
X_train.head()
feature_pipeline(X_test).head()
X_test = feature_pipeline(X_test)
X_test[numerical_vars] = scaler.transform(X_test[numerical_vars])
X_test.head()
def get_keras_data(dataset):
    X = {
        'app': dataset['app'],
        'dev': dataset['device'],
        'os': dataset['os'],
        'ch': dataset['channel'],
        
        'h': dataset['hour'],
        'm': dataset['minute'],
        's': dataset['second'],
        
        'ip_stats': dataset[numerical_vars]
    }
    return X

train_data = get_keras_data(X_train)
val_data = get_keras_data(X_test)
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, concatenate, Dense, Dropout, Lambda
from keras import regularizers
import keras.backend as K

max_app = np.max([X_train['app'].max(), X_test['app'].max()])+1
max_dev = np.max([X_train['device'].max(), X_test['device'].max()])+1
max_os = np.max([X_train['os'].max(), X_test['os'].max()])+1
max_ch = np.max([X_train['channel'].max(), X_test['channel'].max()])+1

max_d = 7
max_h = 24
max_m = 60
max_s = 60

emb_dims = (max_app, max_dev, max_os, max_ch, max_d, max_h, max_m, max_s)

def build_model(emb_dims, optimizer='adam'):
    # ------ Embeddings ------
    max_app, max_dev, max_os, max_ch, max_d, max_h, max_m, max_s = emb_dims
    emb_cat_n = 20
    emb_time_n = 5

    in_app = Input(shape=[1], name = 'app')
    emb_app = Embedding(max_app, emb_cat_n)(in_app)

    in_dev = Input(shape=[1], name = 'dev')
    emb_dev = Embedding(max_dev, emb_cat_n)(in_dev)

    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(max_os, emb_cat_n)(in_os)

    in_ch = Input(shape=[1], name = 'ch')
    emb_ch = Embedding(max_ch, emb_cat_n)(in_ch)

#   ---- if we were using this model in production we'd probably add this back in and train over a larger timeframe ----
#   in_d = Input(shape=[1], name = 'd')
#   emb_d = Embedding(max_d, emb_time_n)(in_d) 

    in_h = Input(shape=[1], name = 'h')
    emb_h = Embedding(max_h, emb_time_n)(in_h) 

    in_m = Input(shape=[1], name = 'm')
    emb_m = Embedding(max_m, emb_time_n)(in_m) 

    in_s = Input(shape=[1], name = 's')
    emb_s = Embedding(max_s, emb_time_n)(in_s) 


    embeddings = concatenate([emb_app, emb_dev, emb_os, emb_ch, emb_h, emb_m, emb_s])
    embeddings_squeeze = Lambda(lambda x: K.squeeze(x, axis=1))(embeddings) # remove sequential dimension

    # ------ Numerical Inputs ------
    in_ip_stats = Input(shape=[len(numerical_vars)], name = 'ip_stats')

    # ------ Model ------
    dense_n = 600
    
    X = concatenate([embeddings_squeeze, in_ip_stats])
    X = Dense(dense_n, activation='relu', kernel_regularizer=regularizers.l2(0.01))(X)
    X = Dropout(0.8)(X)
    pred = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=[in_app, in_dev, in_os, in_ch, in_h, in_m, in_s,
                          in_ip_stats], 
                  outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
model = build_model(emb_dims)
# model.summary()
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-2, decay_factor=0.75, step_size=1)
class_weight = {0:0.20, 1:0.80}
history = model.fit(train_data, y_train, batch_size=2048, epochs=3, 
                    class_weight=class_weight, validation_data=(val_data, y_test), 
                    callbacks=[checkpoint, lr_sched])
model.load_weights('best_weights.h5')
class EnsembleNN:
    def __init__(self, model_builder, weights_paths, emb_dims):
        self.models = []
        for weights in weights_paths:
            model = model_builder(emb_dims)
            model.load_weights(weights)
            self.models.append(model)
            
    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        final_preds = np.mean(preds, axis=0)
        # if we really wanted to maximize performance, we'd weight our predictions instead of taking a simple mean
        return final_preds
class_weight = {0:0.20, 1:0.80}

for i in range(3): 
    print(f'--- Training Model {i+1} of {3} ---')
    model = build_model(emb_dims)
    checkpoint = ModelCheckpoint(f'best_weights_{i}.h5', monitor='val_loss', verbose=0, save_best_only=True)
    history = model.fit(train_data, y_train, batch_size=2048, epochs=3, 
                        class_weight=class_weight, validation_data=(val_data, y_test), 
                        callbacks=[checkpoint, lr_sched])
weights = ['best_weights_0.h5', 'best_weights_1.h5', 'best_weights_2.h5']

ensemble_nn = EnsembleNN(build_model, weights, emb_dims)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
from sklearn.metrics import roc_curve, roc_auc_score

preds = model.predict(val_data)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'Single Model AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
preds = ensemble_nn.predict(val_data)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'Ensemble Model AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
from sklearn.manifold import TSNE

def show_embeddings(layer_weights, name='Embedding'):
    tsne = TSNE(n_components=2, init='pca', random_state=0, method='exact')
    Y = tsne.fit_transform(layer_weights)
    plt.figure(figsize=(8,8))
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.title(name)
show_embeddings(model.layers[7].get_weights()[0], name='App Embeddings')
show_embeddings(model.layers[8].get_weights()[0], name='Device Embeddings')
show_embeddings(model.layers[9].get_weights()[0], name='OS Embeddings')
show_embeddings(model.layers[10].get_weights()[0], name='Channel Embeddings')
show_embeddings(model.layers[11].get_weights()[0], name='Hour Embeddings')
show_embeddings(model.layers[12].get_weights()[0], name='Minute Embeddings')
show_embeddings(model.layers[13].get_weights()[0], name='Second Embeddings')
tsne = TSNE(n_components=2, init='pca', random_state=0, method='exact')
Y = tsne.fit_transform(model.layers[11].get_weights()[0])

annotations = ['night'] * 7 + ['morning'] * 6 + ['afternoon'] * 6 + ['evening'] * 5
colors = [0] * 7 + [1] * 6 + [2] * 6 + [3] * 5

plt.figure(figsize=(8,8))
plt.scatter(Y[:, 0], Y[:, 1], c=colors)
for i, text in enumerate(annotations):
    plt.annotate(text, (Y[i, 0],Y[i, 1]), xytext = (4, 2), textcoords = 'offset points')
plt.title('Hour Embeddings')
tsne = TSNE(n_components=2, init='pca', random_state=0, method='exact')
Y = tsne.fit_transform(model.layers[12].get_weights()[0])

plt.figure(figsize=(8,8))
plt.scatter(Y[:, 0], Y[:, 1], c=range(60))
plt.colorbar()
plt.title('Minute Embeddings')
tsne = TSNE(n_components=2, init='pca', random_state=0, method='exact')
Y = tsne.fit_transform(model.layers[9].get_weights()[0])

from matplotlib.colors import LogNorm
from collections import Counter

# merge dicts to initialize all counts to zero, then fill observed counts keeping others at 0
freq = {**{i: 0 for i in range(max_os)}, **Counter(X_train['os'])} 

plt.figure(figsize=(8,8))
plt.scatter(Y[:, 0], Y[:, 1], c=list(freq.values()), norm=LogNorm())
plt.colorbar()
plt.title('OS Embeddings')
# first, let's clear a few things from memory that we no longer need
import gc
del df
del train_data
del val_data
gc.collect()
from sklearn.preprocessing import OneHotEncoder

categorical_idx = [X_train.columns.get_loc(c) for c in X_train.columns if c in categorical_vars]
enc = OneHotEncoder(categorical_features=categorical_idx)

enc = enc.fit(pd.concat([X_train, X_test]))
X_train_encoded = enc.transform(X_train).tocsr()
X_test_encoded = enc.transform(X_test).tocsr()
X_train_encoded.shape
X_test_encoded.shape
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, concatenate, Dense, Dropout, Lambda
from keras import regularizers
import keras.backend as K

def build_model_onehot(optimizer='adam'):
    # ------ Numerical Inputs ------
    input_layer = Input(shape=[X_train_encoded.shape[1]], name = 'input')

    # ------ Model ------
    X = Dense(450, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    X = Dropout(0.8)(X)
    X = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.01))(X)
    X = Dropout(0.6)(X)
    pred = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=input_layer, 
                  outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
model_onehot = build_model_onehot()
class_weight = {0:0.25,1:0.75}
history = model_onehot.fit(X_train_encoded, y_train, batch_size=1024, epochs=5, 
                           class_weight=class_weight, validation_data=(X_test_encoded, y_test))
from sklearn.metrics import roc_curve, roc_auc_score

preds = model_onehot.predict(X_test_encoded)
fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)

auc = roc_auc_score(y_test, preds)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
plt.title(f'AUC: {auc}')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')