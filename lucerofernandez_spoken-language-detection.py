import numpy as np # linear algebra
np.random.seed(1337) #reproducibility
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import gc
import IPython.display as ipd 
train_path = '../input/spoken-language-identification/train/train/'
test_path = '../input/spoken-language-identification/test/test/'
filename = 'de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.flac'
data, samplerate = sf.read(train_path+filename)
import os
print(os.listdir('../input'))
import pandas as pd
import gc
gc.collect()
data.shape
#el flac cargado
data[:10]
samplerate
ipd.Audio(train_path+filename)
#freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
#plt.pcolormesh(time, freq, Sxx)

## add axis labels
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)

# add axis labels
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.plot(data)

# add axis labels
plt.ylabel('Amplitude')
plt.xlabel('Time in samples')
filename[:2]
#para train path
label = []
for filename in os.listdir(train_path):
    label.append(filename[:2]) #es [:2] porque el idioma esta en los dos primeros elementos
#para test path
label_t = []
for filename in os.listdir(test_path):
    label_t.append(filename[:2])
print(len(label))
print(len(label_t))
# gender = []
# for filename in os.listdir(train_path):
#     gender.append('male' if filename[3:4]=='m' else 'female')
# gender_t = []
# for filename in os.listdir(test_path):
#     gender_t.append('male' if filename[3:4]=='m' else 'female')
file = []
for filename in os.listdir(train_path):
    file.append(filename)
file_t = []
for filename in os.listdir(test_path):
    file_t.append(filename)
Label = pd.DataFrame(label,columns=['Language'])
Label['Language'].value_counts()
data = {'filename':file,
       'languange':label}
data_t = {'filename':file_t,
       'languange':label_t}
#df es el dataframe de train
df = pd.DataFrame(data)
#df_t es el dataframe de test
df_t = pd.DataFrame(data_t)
df['filename'][0]
#extraida de https://github.com/tomasz-oponowicz/spoken_language_identification
def generate_fb_and_mfcc(signal, sample_rate):

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    # Convert from seconds to samples
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate)
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal
    # number of samples without truncating any samples
    # from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40

    low_freq_mel = 0

    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # Convert Mel to Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)

    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks)

    # dB
    filter_banks = 20 * np.log10(filter_banks)

    # MFCCs
    # num_ceps = 12
    # cep_lifter = 22

    # ### Keep 2-13
    # mfcc = dct(
    #     filter_banks,
    #     type=2,
    #     axis=1,
    #     norm='ortho'
    # )[:, 1 : (num_ceps + 1)]

    # (nframes, ncoeff) = mfcc.shape
    # n = np.arange(ncoeff)
    # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    # mfcc *= lift
    #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks
from sklearn.model_selection import train_test_split
### Splitting 73000 audio files to get enough files for training and for RAM
X_train,X_test,y_train,y_test = train_test_split(df,df['languange'],stratify = df['languange'],test_size = 0.5,random_state = 0)
print(X_train['languange'].value_counts())
print(X_test['languange'].value_counts())
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.5,random_state = 0)
print(X_train['languange'].value_counts())
print(X_test['languange'].value_counts())
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.5,random_state = 0)
print(X_train['languange'].value_counts())
print(X_test['languange'].value_counts())
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.6,random_state = 0)
print(X_train['languange'].value_counts())
print(X_test['languange'].value_counts())
X_train['filename'].values[:2]
X_train.head()
#reseteamos los indices
X_train = X_train.reset_index(drop = True)
X_train.head()
gc.collect()
series = []
length = []
for filename in X_train['filename'].values:
    flac, samplerate = sf.read(train_path+filename)
    series.append(flac)
    length.append(samplerate)
X_train['Series'] = series
X_train['Length'] = length
X_train.head(20)
len(X_train)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
##### Clearing the memory and reusing the notebook
gc.collect()
#genero filter banks y mfccs para train
MFCC_array = []
for i in range(0,len(X_train)):
    MFCC = generate_fb_and_mfcc(X_train['Series'][i], X_train['Length'][i])
    MFCC_sc = sc.fit_transform(MFCC)
    MFCC_array.append(MFCC_sc)
MFCC_array = np.array(MFCC_array)  
np.save('../working/MFCC_data',MFCC_array)
#repito todo para test
series_t = []
length_t = []
for filename in df_t['filename'].values:
    flac, samplerate = sf.read(test_path+filename)
    series_t.append(flac)
    length_t.append(samplerate)
df_t['Series'] = series_t
df_t['Length'] = length_t
df_t.head()
##genero filter banks y mfccs para test, el que tiene 540 items
MFCC_array_t = []
for i in range(0,len(df_t)):
    MFCC = generate_fb_and_mfcc(df_t['Series'][i], df_t['Length'][i])
    MFCC_sc = sc.fit_transform(MFCC)
    MFCC_array_t.append(MFCC_sc)
MFCC_array_t = np.array(MFCC_array_t)   
np.save('../working/MFCC_data_t',MFCC_array_t)
#language dummies tiene los one hot encoding para X_train
#language dummies_t tiene los one hot encoding para df_t; i.e. [0, 1, 0], etc
language_dummies = pd.get_dummies(X_train['languange'])
language_dummies_t = pd.get_dummies(df_t['languange'])
np.save('../working/language_dummy',language_dummies.values)
np.save('../working/language_dummy_t',language_dummies_t.values)
import librosa
import librosa.display
#Sample audio feature engineering 
MFCC_array = np.load('../working/MFCC_data.npy')

language_dummies = np.load('../working/language_dummy.npy')
language_dummies_t = np.load('../working/language_dummy_t.npy')
language_dummies_t[:5]
X_train_MFCC,X_test_MFCC,y_train_MFCC,y_test_MFCC = train_test_split(MFCC_array,language_dummies,stratify = language_dummies,test_size = 0.10,random_state = 0)
X_train_MFCC.shape
X_train_MFCC = X_train_MFCC.reshape(-1,1000,40,1)
X_test_MFCC = X_test_MFCC.reshape(-1,1000,40,1)
y_train_MFCC 
y_test_MFCC
len(X_train_MFCC)
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix

from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import Dropout, Input, Activation
from keras.optimizers import Nadam, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

input_shape = (1000,40,1)
model = Sequential()
# model.add(Conv2D(32,(3, 3),strides=(1, 1),padding='same',kernel_regularizer=regularizers.l2(0.0007),
#         input_shape=input_shape,data_format = 'channels_last'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
# model.add(Conv2D(64,(3, 3),strides=(1, 1),padding='same',kernel_regularizer=regularizers.l2(0.0007)))
#         #kernel_regularizer=regularizers.l2(0.001)))
# model.add(Activation('relu'))
# #model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
# model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2),padding='same'))
# model.add(Flatten())
# model.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.0007)))       # kernel_regularizer=regularizers.l2(0.001)))
# model.add(Dropout(0.40))
# model.add(BatchNormalization())
# model.add(Dense(3))
# model.add(Activation('softmax'))

#---------------------------- NEW MODEL

model.add(Conv2D(32,(7, 7), activation='relu', padding='valid', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(64,(5,5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(256,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(512,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


#sgd = SGD(lr=0.01, decay=1, momentum=0.0, nesterov=False)
#sgd = sgd(lr=0.01, decay=1e-6, momentum=0.0, nesterov=False)
#adam = Adam(lr=0.01, decay=1e-6)
import math
from keras.callbacks import LearningRateScheduler
adam = Adam()
def step_decay(epoch):
    # 00158 = 90.4%
	initial_lrate = 0.00158
	drop = 0.9
	epochs_drop = 1
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])



checkpoint = ModelCheckpoint(
                'model.h5',
                monitor='val_acc',
                verbose=0,
                save_best_only=True,
                mode='max'
                )

lrate = LearningRateScheduler(step_decay)
#es = EarlyStopping(monitor='val_loss',mode = 'max')
model.fit(
                X_train_MFCC,
                y_train_MFCC,
                epochs=60,
                callbacks=[checkpoint, lrate],
                verbose=1,
                validation_data=(X_test_MFCC, y_test_MFCC),
                batch_size=32)
model = load_model('model.h5')
model.evaluate(X_test_MFCC,y_test_MFCC)
y_pred = model.predict(X_test_MFCC)
y_test1 = []
for i in range(0,len(y_test_MFCC)):
    argmax = np.argmax(y_test_MFCC[i,:])
    y_test1.append(argmax)
y_pred1 = []
for i in range(0,len(y_test_MFCC)):
    argmax = np.argmax(y_pred[i,:])
    y_pred1.append(argmax)
confusion_matrix(y_test1,y_pred1)
print(classification_report(y_test1,y_pred1))
MFCC_array_t = np.load('../working/MFCC_data_t.npy')
MFCC_array_t.shape
MFCC_array_t = MFCC_array_t.reshape(-1,1000,40,1)
predictions = model.predict(MFCC_array_t)
predictions
y_pred1 = []
for i in range(0,len(predictions)):
    argmax = np.argmax(predictions[i,:])
    y_pred1.append(argmax)
y_test1 = []
for i in range(0,len(language_dummies_t)):
    argmax = np.argmax(language_dummies_t[i,:])
    y_test1.append(argmax)
#confusion_matrix(y_test1,y_pred1)
cm = confusion_matrix(y_test1,y_pred1)
print(np.around(cm/cm.sum(axis=1, keepdims=True)*100,1))
print(classification_report(y_test1,y_pred1))