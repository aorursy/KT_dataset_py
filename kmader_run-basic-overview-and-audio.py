%matplotlib inline
import os, sys
base_data_dir = os.path.join('..', 'input')
sys.path.append(os.path.join(base_data_dir, 'fitparse', 'python-fitparse-master'))
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import fitparse
plt.style.use('ggplot')
experiments_list = {os.path.dirname(x) for x in glob(os.path.join(base_data_dir, '*', '*.fit'))}
experiments_list
test_exp = list(experiments_list)[0]
cur_fit = glob(os.path.join(test_exp, '*.fit'))[0]
cur_csv = glob(os.path.join(test_exp, '*.csv'))[0]
cur_aud = glob(os.path.join(test_exp, '*.wav'))[0]
fit_data = fitparse.FitFile(cur_fit)
fit_df = pd.DataFrame([
    {k['name']: k['value']
     for k in a.as_dict()['fields']} 
    for a in fit_data.get_messages('record')])
fit_df['elapsed_time'] = (fit_df['timestamp']-fit_df['timestamp'].min()).dt.total_seconds()
fit_df.sample(3)
fit_df.plot(x='elapsed_time', y='altitude')
fit_df.plot(x='elapsed_time', y='heart_rate')
sl_df = pd.read_csv(cur_csv)
sl_df = 0.5*(sl_df.fillna(method='backfill')+sl_df.fillna(method='ffill'))
sl_df = sl_df.fillna(method='backfill').fillna(method='ffill')
sl_df.plot(x='relative_time', y='LinearAccelerometerSensor')
sl_df.sample(4)
SAMPLING_RATE = 8000 # [4000, 8000, 16000, 22000]
audio_dat, sr = librosa.core.load(cur_aud, sr=SAMPLING_RATE)
print(audio_dat.shape)
plt.plot(audio_dat)
from scipy.linalg import norm
hop_length = 1000
N_MEL_COUNT = 512
NORMALIZE = False
USE_DB = True
audio_spect = librosa.feature.melspectrogram(audio_dat, sr=sr, hop_length=hop_length, n_mels=N_MEL_COUNT)
if USE_DB:
    audio_spect = librosa.amplitude_to_db(audio_spect, ref=np.max)
if NORMALIZE:
    audio_spect = audio_spect/norm(audio_spect, axis=0, keepdims=True)
print('Spectra per second: {:2.1f}'.format(sr/hop_length))
print(audio_spect.shape)
fig, ax1 = plt.subplots(1, 1, figsize = (20, 5))
plt.colorbar(ax1.imshow(audio_spect, cmap='viridis'))
ax1.set_aspect(3)
audio_time = np.linspace(0, audio_spect.shape[1]/(sr/hop_length), num=audio_spect.shape[1])
from scipy.interpolate import interp1d
DIV_GROUPS = 4
hr_iter_func = interp1d(fit_df['elapsed_time'], fit_df['heart_rate'], 
                        kind='nearest', fill_value='extrapolate')
hr_full_time = hr_iter_func(audio_time)
cat_cut = pd.qcut(hr_full_time, DIV_GROUPS)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
for i in np.unique(cat_cut.codes):
    c_x = cat_cut.codes==i
    ax1.plot(audio_time[c_x], hr_full_time[c_x], '.', label=str(i))
ax1.legend()
train_max = audio_time.shape[0]//2
train_X = audio_spect[:, :train_max].swapaxes(0,1)
train_y = cat_cut.codes[:train_max]
valid_X = audio_spect[:, train_max:].swapaxes(0,1)
valid_y = cat_cut.codes[train_max:]
print('Training:', train_X.shape, train_y.shape)
print('Validation:', valid_X.shape, valid_y.shape)
from keras import layers, models
from keras.optimizers import Adam
def get_simple_mlp():
    simple_model = models.Sequential()
    simple_model.add(layers.BatchNormalization(input_shape=train_X.shape[1:]))
    simple_model.add(layers.GaussianNoise(0.1))
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(128, activation='linear'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(16, activation='linear'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Dense(train_y.max()+1, activation='softmax'))
    simple_model.compile(loss='sparse_categorical_crossentropy',
                        metrics = ['sparse_categorical_accuracy'],
                        optimizer=Adam(lr=1e-4))
    simple_model.summary()
    return simple_model
model_simple = get_simple_mlp()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from IPython.display import clear_output
weight_path="{}_weights.best.hdf5".format('heart_detector')
def fit_model(in_model, batch_size=512):
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=15) # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat]
    fit_results = in_model.fit(train_X, train_y,
                               batch_size=batch_size,
                              epochs=50,
                               shuffle=True,
                              validation_data=(valid_X, valid_y),
                               callbacks=callbacks_list
                              )
    clear_output()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    ax1.plot(fit_results.history['loss'], label='Training')
    ax1.plot(fit_results.history['val_loss'], label='Validation')
    ax1.legend()
    ax1.set_title('Loss')
    b_name = in_model.metrics_names[1]
    ax2.plot(fit_results.history[b_name], label='Training')
    ax2.plot(fit_results.history['val_{}'.format(b_name)], label='Validation')
    ax2.legend()
    ax2.set_title(b_name.replace('_', ' '))
    ax2.set_ylim(0, 1)
    for k, v in zip(in_model.metrics_names,
                    in_model.evaluate(valid_X, valid_y)):
        print('{}: {:2.1f}'.format(k, v))
    return in_model
fit_model(model_simple);
def get_simple_cnn(depth=5):
    simple_model = models.Sequential()
    simple_model.add(layers.BatchNormalization(input_shape=train_X.shape[1:]))
    simple_model.add(layers.GaussianNoise(0.1))
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Reshape(train_X.shape[1:]+(1,)))
    for i in range(depth):
        simple_model.add(layers.Conv1D(4*2**i, 3, activation='linear'))
        simple_model.add(layers.LeakyReLU(0.1))
        simple_model.add(layers.MaxPool1D(2))
    simple_model.add(layers.Flatten())
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(8, activation='linear'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(train_y.max()+1, activation='softmax'))
    simple_model.compile(loss='sparse_categorical_crossentropy',
                        metrics = ['sparse_categorical_accuracy'],
                        optimizer='adam')
    simple_model.summary()
    return simple_model
model_cnn = get_simple_cnn()
fit_model(model_cnn);
from keras.utils import to_categorical
train_max = audio_time.shape[0]//2
train_X = audio_spect[:, :train_max].swapaxes(0,1)
train_y = cat_cut.codes[:train_max]
valid_X = audio_spect[:, train_max:].swapaxes(0,1)
valid_y = cat_cut.codes[train_max:]

def _chunk_it(in_x, raw_y, chunk_size=36):
    out_x, out_y = [], []
    in_y = to_categorical(raw_y)
    for i in range(in_x.shape[0]-chunk_size):
        out_x += [in_x[i:(i+chunk_size)]]
        out_y += [np.mean(in_y[i:(i+chunk_size)], 0)]
    return np.stack(out_x, 0), np.stack(out_y, 0)
train_X, train_y = _chunk_it(train_X, train_y)
valid_X, valid_y = _chunk_it(valid_X, valid_y)
print('Training:', train_X.shape, train_y.shape)
print('Validation:', valid_X.shape, valid_y.shape)
def get_better_cnn(base_count=4, depth=2):
    simple_model = models.Sequential()
    simple_model.add(layers.BatchNormalization(input_shape=train_X.shape[1:]))
    simple_model.add(layers.GaussianNoise(0.1))
    simple_model.add(layers.Dropout(0.25))
    for i in range(depth):
        simple_model.add(layers.Conv1D(base_count*2**i, 3, activation='linear', padding='same'))
        simple_model.add(layers.LeakyReLU(0.1))
        simple_model.add(layers.MaxPool1D(2))
    simple_model.add(layers.Flatten())
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(8, activation='linear'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Dropout(0.25))
    simple_model.add(layers.Dense(train_y.shape[1], activation='softmax'))
    simple_model.compile(loss='categorical_crossentropy',
                        metrics = ['categorical_accuracy'],
                        optimizer='adam')
    simple_model.summary()
    return simple_model
model_bcnn = get_better_cnn(128)
fit_model(model_bcnn);
def get_lstm(base_count=4):
    simple_model = models.Sequential()
    simple_model.add(layers.BatchNormalization(input_shape=train_X.shape[1:]))
    simple_model.add(layers.GaussianNoise(0.2))
    simple_model.add(layers.Dropout(0.75))
    simple_model.add(layers.Conv1D(base_count*2**i, 3, activation='linear', padding='same'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Bidirectional(layers.LSTM(base_count)))
    simple_model.add(layers.Dropout(0.5))
    simple_model.add(layers.Dense(16, activation='linear'))
    simple_model.add(layers.LeakyReLU(0.1))
    simple_model.add(layers.Dropout(0.5))
    simple_model.add(layers.Dense(train_y.shape[1], activation='softmax'))
    simple_model.compile(loss='categorical_crossentropy',
                        metrics = ['categorical_accuracy'],
                        optimizer=Adam(1e-4))
    simple_model.summary()
    return simple_model
model_lstm = get_lstm(128)
fit_model(model_lstm);
