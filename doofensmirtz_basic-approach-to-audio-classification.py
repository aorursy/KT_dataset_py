import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib notebook



import os

import glob

from tqdm import tqdm_notebook



import librosa

import librosa.display



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical
!ls ../input/environmental-sound-classification-50/
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
DF_PATH = "/kaggle/input/environmental-sound-classification-50/esc50.csv"

S44_DIR = "/kaggle/input/environmental-sound-classification-50/audio/audio/44100"

S16_DIR = "/kaggle/input/environmental-sound-classification-50/audio/audio/16000"
class conf:

    # Preprocessing settings

    sampling_rate = 44100

    duration = 2

    hop_length = 347*duration # to make time steps 128

    fmin = 20

    fmax = sampling_rate // 2

    n_mels = 128

    n_fft = n_mels * 20

    samples = sampling_rate * duration
def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 

                             sr=conf.sampling_rate, hop_length=conf.hop_length,

                            fmin=conf.fmin, fmax=conf.fmax)

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.show()



def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):

    x, sr = librosa.load(pathname , sr = conf.sampling_rate)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

    return mels
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

    X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    std = std or X.std()

    Xstd = (X - mean) / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Scale to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V



def convert_wav_to_image(df, source, fold):

    X = []

    y = []

    

    temp_df = df[df['fold'].isin(fold)]

    for i, row in tqdm_notebook(temp_df.iterrows()):

        x = read_as_melspectrogram(conf, os.path.join(source , row.filename), trim_long_data=False)

        x_color = mono_to_color(x)

        X.append(x_color)

        y.append(row.target)

    return X, y
df = pd.read_csv(DF_PATH)

df
df_10 = df[df['esc10'] == True]

df_10
classes = df_10.category.unique()



class_dict = {x:i for i,x in enumerate(classes)}

df_10['target'] = df['category'].map(class_dict)

df_10['target'] = df_10['target'].astype('int32')
df_10
x_train , y_train = convert_wav_to_image(df_10 , S44_DIR  ,fold = [1,2,3,4])

x_val , y_val = convert_wav_to_image(df_10 , S44_DIR , fold = [5])
x_train , x_val  = np.array(x_train), np.array(x_val)
x_train.shape
y_train , y_val = to_categorical(y_train, num_classes=10) , to_categorical(y_val, num_classes=10)
def get_model(f1=3,f2=3):    

    

    model = keras.Sequential([

                                layers.Conv2D(16, (f1,f2), padding= 'same', activation='relu', input_shape=(128,318,3) ),

                                layers.MaxPooling2D(2, padding='same'),

                                

                                layers.Conv2D(32, (f1,f2), padding= 'same', activation='relu'),

                                layers.MaxPooling2D(2, padding='same'),

                                layers.Dropout(0.3),



                                layers.Conv2D(64, (f1,f2), padding= 'same', activation='relu'),

                                layers.MaxPooling2D(2, padding='same'),

                                layers.Dropout(0.3),

            

                                layers.Conv2D(128, (f1,f2),padding='same', activation = 'relu'),

                                layers.MaxPooling2D(2 ,padding='same'),

                                layers.Dropout(0.3),

            

                                layers.GlobalAveragePooling2D(),

            

                                layers.Dense(10, activation='softmax')

   

                                ])

    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])

    model.summary()



    return model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
model_1 = get_model()
history = model_1.fit(x_train ,y_train , epochs=50, batch_size =32 , shuffle=True, validation_data=(x_val,y_val), callbacks=[callback])