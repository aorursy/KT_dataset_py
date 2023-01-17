# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

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

import numpy as np
import gc

gc.collect()
data.shape
data[:10]
samplerate
ipd.Audio(train_path+filename)
freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')

plt.pcolormesh(time, freq, Sxx)



# add axis labels

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')
Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)



# add axis labels

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')
plt.plot(data)
label = []

for filename in os.listdir(train_path):

    label.append(filename[:2])
label_t = []

for filename in os.listdir(test_path):

    label_t.append(filename[:2])
len(label_t)
gender = []

for filename in os.listdir(train_path):

    gender.append('male' if filename[3:4]=='m' else 'female')
gender_t = []

for filename in os.listdir(test_path):

    gender_t.append('male' if filename[3:4]=='m' else 'female')
file = []

for filename in os.listdir(train_path):

    file.append(filename)
file_t = []

for filename in os.listdir(test_path):

    file_t.append(filename)
Label = pd.DataFrame(label,columns=['Language'])
Label['Language'].value_counts()
data = {'Gender':gender,

        'filename':file,

       'languange':label}
data_t = {'Gender':gender_t,

        'filename':file_t,

       'languange':label_t}
df = pd.DataFrame(data)
df_t = pd.DataFrame(data_t)
df['filename'][0]
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



    return filter_banks
from sklearn.model_selection import train_test_split
### Splitting 73000 audio files to get 6000 files for training
X_train,X_test,y_train,y_test = train_test_split(df,df['languange'],stratify = df['languange'],test_size = 0.5,random_state = 0)
X_train['languange'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.5,random_state = 0)
X_train['languange'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.5,random_state = 0)
X_train['languange'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X_train,X_train['languange'],stratify = X_train['languange'],test_size = 0.6,random_state = 0)
X_train['languange'].value_counts()
X_train['filename'].values[:2]
X_train.head()
X_train = X_train.reset_index(drop = True)
X_train.head()
series = []

length = []

for filename in X_train['filename'].values:

    flac, samplerate = sf.read(train_path+filename)

    series.append(flac)

    length.append(samplerate)
X_train['Series'] = series
X_train['Length'] = length
X_train.head(100)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
##### Clearing the memory and reusing the notebook
MFCC_array = []

for i in range(0,len(X_train)):

    MFCC = generate_fb_and_mfcc(X_train['Series'][i], X_train['Length'][i])

    MFCC_sc = sc.fit_transform(MFCC)

    MFCC_array.append(MFCC_sc)

MFCC_array = np.array(MFCC_array)  
np.save('../input/MFCC_data',MFCC_array)
np.save('MFCC_data',MFCC_array)
series_t = []

length_t = []

for filename in df_t['filename'].values:

    flac, samplerate = sf.read(test_path+filename)

    series_t.append(flac)

    length_t.append(samplerate)
df_t['Series'] = series_t
df_t['Length'] = length_t
df_t.head()
MFCC_array_t = []

for i in range(0,len(df_t)):

    MFCC = generate_fb_and_mfcc(df_t['Series'][i], df_t['Length'][i])

    MFCC_sc = sc.fit_transform(MFCC)

    MFCC_array_t.append(MFCC_sc)

MFCC_array_t = np.array(MFCC_array_t)   
np.save('../input/MFCC_data_t',MFCC_array_t)
np.save('MFCC_data_t',MFCC_array_t)
language_dummies = pd.get_dummies(X_train['languange'])

language_dummies_t = pd.get_dummies(df_t['languange'])

np.save('../input/language_dummy',language_dummies.values)

np.save('../input/language_dummy_t',language_dummies_t.values)
np.save('language_dummy',language_dummies.values)

np.save('language_dummy_t',language_dummies_t.values)
#### Not using this function

def wav2mfcc(wave, max_len=1000):

#     mfcc = librosa.feature.mfcc(wave, sr=16000)

    mfcc = librosa.feature.mfcc(wave, n_mfcc=13)



    # If maximum length exceeds mfcc lengths then pad the remaining ones

    if (max_len > mfcc.shape[1]):

        pad_width = max_len - mfcc.shape[1]

        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')



    # Else cutoff the remaining parts

    else:

        mfcc = mfcc[:, :max_len]

    

    return mfcc
import librosa

import librosa.display
#Sample audio feature engineering 
data_proc = generate_fb_and_mfcc(data, samplerate)
data_proc.shape
np.save('../input/data_proc',data_proc)
np.load('../input/data_proc.npy')
MFCC_array = np.load('../input/MFCC_data.npy')

language_dummies = np.load('../input/language_dummy.npy')

language_dummies_t = np.load('../input/language_dummy_t.npy')
language_dummies_t
X_train_tf,X_test_tf,y_train_tf,y_test_tf = train_test_split(MFCC_array,language_dummies,stratify = language_dummies,test_size = 0.10,random_state = 0)
X_train_tf.shape
X_train_tf = X_train_tf.reshape(-1,1000,40,1)
X_test_tf = X_test_tf.reshape(-1,1000,40,1)
y_train_tf 

y_test_tf 
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

model.add(Conv2D(

        32,

        (3, 3),

        strides=(1, 1),

        padding='same',

        #kernel_regularizer=regularizers.l2(0.001),

        input_shape=input_shape,data_format = 'channels_last'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

model.add(Conv2D(

        64,

        (3, 3),

        strides=(1, 1),

        padding='same'))

        #kernel_regularizer=regularizers.l2(0.001)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

model.add(AveragePooling2D(

        pool_size=(3, 3),

        strides=(2, 2),

        padding='same'))

model.add(Flatten())



model.add(Dense(

        128,

        activation='relu'))

       # kernel_regularizer=regularizers.l2(0.001)))



model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(Dense(3))

model.add(Activation('softmax'))



#sgd = sgd(lr=0.01, decay=1e-6, momentum=0.0, nesterov=False)

#adam = Adam(lr=0.01, decay=1e-6)

model.compile(

        loss='categorical_crossentropy',

        optimizer='Adadelta',

        metrics=['accuracy'])

##### Do not run this

# Convolution layer

classifier = Sequential()

classifier.add(Conv2D(128,(2,2),strides=(1,1),padding='same',input_shape=(1000,40,1),activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(units = 64,activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.20))

classifier.add(Dense(units=3,activation='softmax'))

#optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)

classifier.compile(optimizer='Adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
checkpoint = ModelCheckpoint(

                'model.h5',

                monitor='val_loss',

                verbose=0,

                save_best_only=True,

                mode='min')

#es = EarlyStopping(monitor='val_loss',mode = 'max')

model.fit(

                X_train_tf,

                y_train_tf,

                epochs=100,

                callbacks=[checkpoint],

                verbose=1,

                validation_data=(X_test_tf, y_test_tf),

                batch_size=32)
model.save('../input/Model_CNN.h5')
model.save('Model_CNN.h5')
y_pred = model.predict(X_test_tf)
y_test1 = []

for i in range(0,len(y_test_tf)):

    argmax = np.argmax(y_test_tf[i,:])

    y_test1.append(argmax)
y_pred1 = []

for i in range(0,len(y_test_tf)):

    argmax = np.argmax(y_pred[i,:])

    y_pred1.append(argmax)
confusion_matrix(y_test1,y_pred1)
print(classification_report(y_test1,y_pred1))
MFCC_array_t = np.load('../input/MFCC_data_t.npy')
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
confusion_matrix(y_test1,y_pred1)
print(classification_report(y_test1,y_pred1))