!du -sh ../input/*
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
from IPython.display import Audio
file = '786ee883.wav'
path = '../input/audio_train/audio_train/'
Audio(filename=path+file)
# Vamos a definir una funcion para extraer la duracion de un audio en segundos
import wave

def get_length(file):
    audio = wave.open(path+file)
    return audio.getnframes() / audio.getframerate()

get_length(file)
# Vamos a procesar en paralelo la funcion en todos los archivos
from joblib import Parallel, delayed

with Parallel(n_jobs=10, prefer='threads', verbose=1) as ex:
    lengths = ex(delayed(get_length)(e) for e in df.fname)
df['length'] = lengths
df.head()
df = df.query('length <= 6').reset_index(drop=True)
print(df.shape)
df.head()
import librosa

y, sr = librosa.load(path+file)
# y : audio data
# sr: sample rate

plt.plot(y)
plt.title(f'Sample rate = {sr}', size=18);
# Ahora obtengamos la representacion MFCC
mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40)
print(mfcc.shape)

plt.figure(figsize=(10,5))
plt.imshow(mfcc, cmap='hot');
# Definimos una funcion para obtener los features
def obtain_mfcc(file, features=40):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    return librosa.feature.mfcc(y, sr, n_mfcc=features)
obtain_mfcc(file).shape
mfcc.shape
def get_mfcc(file, n_mfcc=40, padding=None):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    if padding: mfcc = np.pad(mfcc, ((0, 0), (0, max(0, padding-mfcc.shape[1]))), 'constant')
    return mfcc.astype(np.float32)

mfcc = get_mfcc(file, padding=200)
print(mfcc.shape)
plt.figure(figsize=(12,5))
plt.imshow(mfcc, cmap='hot');
# Veamos cuanto padding necesitamos para el archivo de mayor duracion
print(get_mfcc(df.sort_values('length').fname.iloc[-1]).shape)
from functools import partial

n_mfcc = 40
padding = 259
fun = partial(get_mfcc, n_mfcc=n_mfcc, padding=padding)

with Parallel(n_jobs=10, prefer='threads', verbose=1) as ex:
    mfcc_data = ex(delayed(partial(fun))(e) for e in df.fname)
    
# Juntamos la data en un solo array y agregamos una dimension
mfcc_data = np.stack(mfcc_data)[..., None]
mfcc_data.shape
lbl2idx = {lbl:idx for idx,lbl in enumerate(df.label.unique())}
idx2lbl = {idx:lbl for lbl,idx in lbl2idx.items()}
n_categories = len(lbl2idx)
lbl2idx
n_categories = len(lbl2idx)
df['y'] = df.label.map(lbl2idx)
df.head()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(mfcc_data, df.y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Input, GlobalAvgPool2D, GlobalMaxPool2D, concatenate
from keras.optimizers import Adam, SGD
import keras.backend as K
bs = 128
lr = 0.003

m_in = Input([n_mfcc, padding, 1])
x = BatchNormalization()(m_in)

layers = [10, 20, 50, 100]
for i,l in enumerate(layers):
    strides = 1 if i == 0 else (2,2)
    x = Conv2D(l, 3, strides=strides, activation='relu', padding='same',
               use_bias=False, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.02)(x)

x_avg = GlobalAvgPool2D()(x)
x_max = GlobalMaxPool2D()(x)

x = concatenate([x_avg, x_max])
x = Dense(1000, activation='relu', use_bias=False, kernel_initializer='he_uniform')(x)
x = Dropout(0.2)(x)
m_out = Dense(n_categories, activation='softmax')(x)

model = Model(m_in, m_out)
model.compile(Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
log1 = model.fit(x_train, y_train, bs, 15, validation_data=[x_val, y_val])
K.eval(model.optimizer.lr.assign(lr/10))
log2 = model.fit(x_train, y_train, bs, 10, validation_data=[x_val, y_val])
def show_results(*logs):
    trn_loss, val_loss, trn_acc, val_acc = [], [], [], []
    
    for log in logs:
        trn_loss += log.history['loss']
        val_loss += log.history['val_loss']
        trn_acc += log.history['acc']
        val_acc += log.history['val_acc']
    
    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    ax1, ax2 = axes
    ax1.plot(trn_loss, label='train')
    ax1.plot(val_loss, label='validation')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')
    ax2.plot(trn_acc, label='train')
    ax2.plot(val_acc, label='validation')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy')
    for ax,title in zip(axes, ['Train', 'Accuracy']):
        ax.set_title(title, size=14)
        ax.legend()
show_results(log1, log2)
sample = df.sample()
sample_file = sample.fname.iloc[0]
sample_label = sample.label.iloc[0]

mfcc = get_mfcc(sample_file, n_mfcc, padding)[None, ..., None]
y_ = model.predict(mfcc)
pred = idx2lbl[np.argmax(y_)]

print(f'True       = {sample_label}')
print(f'Prediction = {pred}')
Audio(path + sample_file)