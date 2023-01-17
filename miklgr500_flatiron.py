import os
import gc
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from scipy import signal
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from numba import jit, int32
INIT_DIR = '../input'
SIZE = 1024
os.listdir(INIT_DIR)
meta = pd.read_csv(os.path.join(INIT_DIR, 'metadata_train.csv'))
train = pq.read_pandas(os.path.join(INIT_DIR, 'train.parquet'), columns=[str(i) for i in range(1000)]).to_pandas()
meta.describe()
meta.corr()
positive_mid = np.unique(meta.loc[meta.target == 1, 'id_measurement'].values)
negative_mid = np.unique(meta.loc[meta.target == 0, 'id_measurement'].values)
pid = meta.loc[meta.id_measurement == positive_mid[0], 'signal_id']
nid = meta.loc[meta.id_measurement == negative_mid[0], 'signal_id']
positive_sample = train.iloc[:, pid]
negative_sample = train.iloc[:, nid]
@jit('float32(float32[:,:], int32, int32)')
def flatiron(x, alpha=50., beta=1):
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha
        new_x[i] =  x[i] - zero
    return new_x
plt.figure(figsize=(24, 8))
plt.plot(positive_sample, alpha=0.8);
x_filt = flatiron(positive_sample.values)
plt.figure(figsize=(24, 8))
plt.plot(x_filt, alpha=0.5);
plt.figure(figsize=(24, 8))
plt.plot(negative_sample, alpha=0.7);
x_filt = flatiron(negative_sample.values)
plt.figure(figsize=(24, 8))
plt.plot(x_filt, alpha=0.5);
@jit('float32(float32[:,:], int32, int32)')
def feature_extractor(x, n_part=1000, n_dim=3):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part, n_dim))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output

@jit('float32(float32[:,:])')
def basic_feature_extractor(x):
    return [np.max(x, axis=0), np.min(x, axis=0), np.std(x, axis=0), np.sum(x, axis=0)/len(x)]
x_train = []
basic = []
y_basic = []
y_train = []
mid = np.unique(meta.id_measurement.values)
for b in tqdm(range(4)):
    start = b*len(meta)//12
    if len(meta)//3 - start < len(meta)//12:
        end = -1
    else:
        end = start + len(meta)//12
    
    columns = []
    for i in mid[start:end]:
        columns.extend(meta.loc[meta.id_measurement==i, 'signal_id'].values.tolist())
    train = pq.read_pandas(os.path.join(INIT_DIR, 'train.parquet'), columns=[str(i) for i in columns]).to_pandas()
    
    for i in range(len(train.columns)):
        train.iloc[:, i] = flatiron(train.iloc[:, i].values)
        
    for i in mid[start:end]:
            idx = meta.loc[meta.id_measurement==i, 'signal_id'].values
            x_train.append(abs(feature_extractor(train.loc[:, [str(kj) for kj in idx]].values, n_part=SIZE)))
            basic.extend([basic_feature_extractor(train.loc[:, str(kj)].values) for kj in idx])
            y_basic.extend([meta.loc[meta.signal_id==kj, 'target'].values for kj in idx])
            y_train.append(meta.loc[meta.id_measurement==i, 'target'].values)
del train;gc.collect()
np.shape(basic)
x_base = np.array(basic)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_base)
y_basic = np.array(y_basic)
x_pos, x_neg = [], []
for _x, _y in zip(x_pca, y_basic):
    if _y == 1:
        x_pos.append(_x)
    else:
        x_neg.append(_x)
x_pos, x_neg = np.array(x_pos), np.array(x_neg)
x_pos.shape
plt.figure(figsize=(10, 10))
plt.scatter(x_neg[:, 0], x_neg[:, 1])
plt.scatter(x_pos[:, 0], x_pos[:, 1])
np.unique(y_train)
x_train = np.array(x_train)
y_train = np.array(y_train)
print(np.shape(x_train), np.shape(y_train))
csum = np.sum(y_train, axis=-1)
np.unique(csum)
pos_index = np.where(csum>0)[0]
neg_index = np.where(csum==0)[0]
figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24,8))
sns.heatmap(x_train[pos_index[0], :, :], ax=ax1)
sns.heatmap(x_train[pos_index[10], :, :], ax=ax2)
sns.heatmap(x_train[pos_index[20], :, :], ax=ax3)
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off();
figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24,8))
sns.heatmap(x_train[neg_index[0], :, :], ax=ax1)
sns.heatmap(x_train[neg_index[10], :, :], ax=ax2)
sns.heatmap(x_train[neg_index[20], :, :], ax=ax3)
ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off();
from keras.layers import *
from keras import Model
from keras.optimizers import Nadam
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
import keras.backend as K
class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size=64):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        
    def __getitem__(self, index):
        index = np.random.choice([i for i in range(len(self._y))], size=(self._batch_size))
        
        x_batch, y_batch, w = [], [], []
        for _x, _y in zip(self._x[index], self._y[index]):
            _x = self.cyclic_shift(_x)
            #_x, _y = self.phase_permutation(_x, _y)
            x_batch.append(_x)
            y_batch.append(_y)
            #w.append(np.sum(_y)*100+1)
        return np.array(x_batch), np.array(y_batch)#, np.array(w)
    
    def __len__(self):
        return len(self._y)//self._batch_size
    
    @staticmethod
    def cyclic_shift(x, alpha=0.5):
        s = np.random.uniform(0, alpha)
        part = int(len(x)*s)
        x_ = x[:part, :]
        _x = x[-len(x)+part:, :]
        return np.concatenate([_x, x_], axis=0)
    
    @staticmethod
    def phase_permutation(x, y):
        phase = np.random.permutation([0,1,2])
        out_x, out_y = [], []
        for indx in phase:
            out_x.append(x[..., indx])
            out_y.append(y[indx])
        return np.stack(out_x, axis=-1), np.array(out_y)
def matthews_corr_coeff(y_true, y_pred):
    y_pos_pred = K.round(K.clip(y_pred, 0, 1))
    y_pos_true = K.round(K.clip(y_true, 0, 1))
    
    y_neg_pred = 1 - y_pos_pred
    y_neg_true = 1 - y_pos_true

    tp = K.sum(y_pos_true * y_pos_pred)
    tn = K.sum(y_neg_true * y_neg_pred)
    fp = K.sum(y_neg_true * y_pos_pred)
    fn = K.sum(y_pos_true * y_neg_pred)
    return (tp * tn - fp * fn) / (K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + K.epsilon())
def get_model(inp_shape=(SIZE, 3)):
    inp = Input(inp_shape)
    # 256
    x = Conv1D(32, kernel_size=3, dilation_rate=3,use_bias=False,kernel_constraint=max_norm(2.))(inp)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 250
    x = Conv1D(32, kernel_size=3, dilation_rate=2,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 245
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    
    #  122
    x = Conv1D(64, kernel_size=3, dilation_rate=3,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 116
    x = Conv1D(64, kernel_size=3, dilation_rate=2,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 112
    x = Conv1D(64, kernel_size=3, dilation_rate=1,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 110
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    
    # 54
    x = Conv1D(128, kernel_size=3, dilation_rate=3,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 48
    x = Conv1D(128, kernel_size=3, dilation_rate=2,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 44
    x = Conv1D(128, kernel_size=3, dilation_rate=1,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 42
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    
    #  20
    x = Conv1D(256, kernel_size=3, dilation_rate=3,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 15
    x = Conv1D(256, kernel_size=3, dilation_rate=2,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 10
    x = Conv1D(256, kernel_size=3, dilation_rate=1,use_bias=False,kernel_constraint=max_norm(2.))(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    # 8
    x = GlobalMaxPooling1D()(x)
    
    x = Dropout(0.75)(x)
    
    max_out = []
    for _ in range(5):
        max_out.append(Dense(128,use_bias=False,kernel_constraint=max_norm(2.))(x))
    x = Maximum()(max_out)
    x = BatchNormalization()(x)
    
    out = Dense(3, activation='sigmoid',kernel_constraint=max_norm(2.))(x)
    return Model(inp, out)
mcp = ModelCheckpoint('model.h5',monitor='val_matthews_corr_coeff', mode='max')
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, shuffle=True, train_size=0.75, random_state=28, stratify=csum)
tr_gen = DataGenerator(x_tr, y_tr)
vl_gen = DataGenerator(x_val, y_val)
model = get_model()
print(model.summary())
model.compile(optimizer=Nadam(4*1e-3, schedule_decay=1e-7),loss='binary_crossentropy', metrics=['accuracy', matthews_corr_coeff])
model.fit_generator(tr_gen, steps_per_epoch=1000, epochs=10, callbacks=[mcp], validation_data=vl_gen, validation_steps=400)
model.load_weights('model.h5')
best_thr = 0.01
best_metric = 0
y_val = y_val.flatten()
proba = model.predict(x_val)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=28)
for i in tqdm(np.linspace(0.01, 0.9999, 30)):
    for j in np.linspace(0.01, 0.9999, 30):
        for k in np.linspace(0.01, 0.9999, 30):
            y_pred = np.int32(np.stack([proba[:, 0] > i, proba[:, 1] > j, proba[:, 2] > k], axis=-1)).flatten()
            m = matthews_corrcoef(y_val, y_pred)
            if m > best_metric:
                best_thr = (i, j, k)
                best_metric= m
print('Best threshold: ',best_thr, ' ; Best metric: ',best_metric)
meta = pd.read_csv(os.path.join(INIT_DIR, 'metadata_test.csv'))
submission = pd.read_csv(os.path.join(INIT_DIR, 'sample_submission.csv'))
meta.corr()
len(meta.id_measurement.unique())*3
len(meta.signal_id)
for b in tqdm(range(0, len(meta), 3000)):
    idx = []
    if b+3000 < len(meta):
        idx = meta.signal_id[b:b+3000].values
    else:
        idx = meta.signal_id[b:].values
    subset_test = pq.read_pandas(os.path.join(INIT_DIR, 'test.parquet'), columns=[str(j) for j in idx]).to_pandas()
    x_batch = []
    for i in range(0, len(idx)//3):
        _x  = []
        for j in range(0, 3):
            _x.append(flatiron(subset_test.iloc[:, i*3+j].values))
        _x = np.concatenate(_x, axis=-1)
        x_batch.append(feature_extractor(_x, n_part=SIZE))
    y_batch = model.predict(np.array(x_batch), verbose=0)
    pred = []
    for yj in y_batch:
        for j, yi in enumerate(yj):
            pred.append(np.int32(yi > best_thr[j]))
    for jdx, iy in zip(idx, pred):
        submission.loc[submission.signal_id == jdx, 'target'] = iy
submission.to_csv('submission.csv', index=False)
submission.head()

