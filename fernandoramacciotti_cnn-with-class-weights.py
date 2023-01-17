import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

import pyarrow.parquet as pq
from numba import jit, njit
import dask.dataframe as dd
import dask.array as da
from tqdm import tqdm
import gc

%matplotlib inline
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
def mcc(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
%%time
train_df = pq.read_pandas('../input/train.parquet').to_pandas()
meta_train = pd.read_csv('../input/metadata_train.csv')
@jit('float32[:, :](float32[:,:], int32)')
def feature_extractor(x, n_part=1000,):
    lenght = len(x)
    n_feat = 7
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part, n_feat))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j, 0] = np.mean(k, axis=0) #mean
        output[j, 1] = np.min(k, axis=0) #min
        output[j, 2] = np.max(k, axis=0) #max
        output[j, 3] = np.std(k, axis=0) #std
        output[j, 4] = np.median(k, axis=0) #median
        output[j, 5] = stats.skew(k, axis=0) #skew
        output[j, 6] = stats.kurtosis(k, axis=0) # kurtosis
    return output
X = []
y = []
for i in tqdm(meta_train.signal_id):
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    y.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)
    X.append(feature_extractor(train_df.iloc[:, idx].values, n_part=400))
del train_df; gc.collect()
X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1])
y = np.array(y).reshape(-1,)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
(train_idx, val_idx) = next(sss.split(X, y))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
#X_train, X_val, y_train, y_val = train_test_split(
#    X, y, stratify=y, test_size=0.3, shuffle=True, random_state=0)

scalers = {}
for i in range(X_train.shape[2]):
    scalers[i] = MinMaxScaler(feature_range=(-1, 1))
    X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :]) 

for i in range(X_val.shape[2]):
    X_val[:, i, :] = scalers[i].transform(X_val[:, i, :]) 
def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
model = keras.Sequential([
    Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Conv1D(filters=64, kernel_size=4, activation='relu'),
    MaxPooling1D(2),
    Conv1D(filters=20, kernel_size=4, activation='relu'),
    Conv1D(filters=20, kernel_size=4, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='tanh'),
    Dropout(0.5),
    Dense(8, activation='tanh'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[mcc, keras_auc])
from sklearn.utils import class_weight
epochs = 30
batch_size = 16

#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_train),
#                                                 y_train)

class_weights = {
    0: 1.,
    1: 1.2,
}

model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=epochs, batch_size=batch_size,
          class_weight=class_weights)
del X, X_train, X_val; gc.collect()
meta_test = pd.read_csv('../input/metadata_test.csv')
start_test = meta_test.signal_id.min()
end_test = meta_test.signal_id.max()
X_test = []
y_test = []

pool_test = 2000

for start_col in tqdm(range(start_test, end_test + 1, pool_test)):
    end_col = min(start_col + pool_test, end_test + 1)
    print('cols {}-{}'.format(start_col, end_col-1))
    test = pq.read_pandas('../input/test.parquet',
                          columns=[str(c) for c in range(start_col, end_col)]).to_pandas()
    print(test.shape)
    for i in tqdm(test.columns):
        X_test.append(feature_extractor(test[i].values, n_part=400))
        test.drop([i], axis=1, inplace=True); gc.collect()
    del test; gc.collect()
X_test = np.array(X_test).reshape(-1, X_test[0].shape[0], X_test[0].shape[1])
for i in range(X_test.shape[2]):
    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
    
y_pred = model.predict_classes(X_test)
submission = pd.read_csv('../input/sample_submission.csv')
submission['signal_id'] = meta_test.signal_id.values
submission['target'] = y_pred.astype(int)
submission.to_csv('submission.csv', index=False)