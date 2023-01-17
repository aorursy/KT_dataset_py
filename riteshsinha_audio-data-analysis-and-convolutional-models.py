import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

import os

import IPython

import IPython.display

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from PIL import Image               # to load images

from IPython.display import display

import time

print(os.listdir("../input"))
# Special thanks to https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py

class EasyDict(dict):

    def __init__(self, d=None, **kwargs):

        if d is None:

            d = {}

        if kwargs:

            d.update(**kwargs)

        for k, v in d.items():

            setattr(self, k, v)

        # Class attributes

        for k in self.__class__.__dict__.keys():

            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):

                setattr(self, k, getattr(self, k))



    def __setattr__(self, name, value):

        if isinstance(value, (list, tuple)):

            value = [self.__class__(x)

                     if isinstance(x, dict) else x for x in value]

        elif isinstance(value, dict) and not isinstance(value, self.__class__):

            value = self.__class__(value)

        super(EasyDict, self).__setattr__(name, value)

        super(EasyDict, self).__setitem__(name, value)



    __setitem__ = __setattr__



    def update(self, e=None, **f):

        d = e or dict()

        d.update(f)

        for k in d:

            setattr(self, k, d[k])



    def pop(self, k, d=None):

        delattr(self, k)

        return super(EasyDict, self).pop(k, d)
conf = EasyDict()

conf.sampling_rate = 44100

conf.duration = 2

conf.hop_length = 347 # to make time steps 128

conf.fmin = 20

conf.fmax = conf.sampling_rate // 2

conf.n_mels = 128

conf.n_fft = conf.n_mels * 20

conf.samples = conf.sampling_rate * conf.duration
import librosa

import librosa.display

def read_audio(conf, pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > conf.samples: # long enough

        if trim_long_data:

            y = y[0:0+conf.samples]

    else: # pad blank

        padding = conf.samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y
start_time_data_processing = time.time()



DATA = Path('../input')

CSV_TRN_CURATED = DATA/'train_curated.csv'

CSV_TRN_NOISY = DATA/'train_noisy.csv'

CSV_SUBMISSION = DATA/'sample_submission.csv'

TRN_CURATED = DATA/'train_curated'

TRN_NOISY = DATA/'train_noisy'

TEST = DATA/'test'



WORK = Path('work')

IMG_TRN_CURATED = WORK/'image/trn_curated'

IMG_TRN_NOISY = WORK/'image/train_noisy'

IMG_TEST = WORK/'image/test'



df_train_curated = pd.read_csv(CSV_TRN_CURATED)

print(df_train_curated.head(10))

# Collecting various data frames for further processing.

df_bark = df_train_curated.loc[df_train_curated['labels'] == 'Bark'][1:5]

df_run = df_train_curated.loc[df_train_curated['labels'] == 'Run'][1:5]
buzz_1_file =  DATA/'train_curated'/'02f54ef1.wav'

y_2_secs = read_audio(conf, buzz_1_file, trim_long_data = True)

y_full = read_audio(conf, buzz_1_file, trim_long_data = False)

print(len(y_full))

print(len(y_2_secs))

print(y_full.shape[0]/44100)
bark_file = DATA/'train_curated'/'0006ae4e.wav'

y_bark = read_audio(conf, buzz_1_file, trim_long_data = False)

print(y_bark.shape[0]/ 44100)
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

    x = read_audio(conf, pathname, trim_long_data)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

    return mels

TRN_CURATED = DATA/'train_curated'

x = read_audio(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False)

print(x.shape)

mels = audio_to_melspectrogram(conf, x)

print(mels.dtype)

print(mels.shape)

print(mels)
bark = read_as_melspectrogram(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False, debug_display=True)

buzz = read_as_melspectrogram(conf, TRN_CURATED/'02f54ef1.wav', trim_long_data=False, debug_display=True)
print(bark.shape)

print(buzz.shape)
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



def convert_wav_to_image(df, source, img_dest):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)

        x_color = mono_to_color(x)

        X.append(x_color)

    return df, X



def convert_wav_to_cropped_image(df, source, img_dest):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)

        x_color = mono_to_color(x)

        # - - - - - - - - - - #

        x_color = Image.fromarray(x_color)

        time_dim, base_dim = x_color.size

        crop_x = random.randint(0, time_dim - base_dim)

        x_cropped = x_color.crop([crop_x, 0, crop_x+base_dim, base_dim]) 

        # - - - - - - - - - - #

        X.append(x_cropped)

    return df, X
print(bark.shape)

bark_image = mono_to_color(bark)

print(bark_image.shape)
import random

x = Image.fromarray(bark_image)

display(x)

time_dim, base_dim = x.size

crop_x = random.randint(0, time_dim - base_dim)

x_cropped = x.crop([crop_x, 0, crop_x+base_dim, base_dim]) 

display(x_cropped)
def get_cropped_image(conf, path_of_image, display_image =True):

    mel_spec_gram = read_as_melspectrogram(conf, path_of_image, trim_long_data=False, debug_display=False)

    img_array = mono_to_color(mel_spec_gram)

    img = Image.fromarray(img_array)

    time_dim, base_dim = img.size

    cropped = random.randint(0, time_dim - base_dim)

    cropped_image = img.crop([cropped, 0, cropped+base_dim, base_dim]) 

    if display:

        display(cropped_image)

    return(cropped_image)
x1 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[0,0])

x2 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[1,0])

x3 = get_cropped_image(conf, TRN_CURATED/df_bark.iloc[2,0])
get_cropped_image(conf, TRN_CURATED/df_run.iloc[0,0])

get_cropped_image(conf, TRN_CURATED/df_run.iloc[1,0])

get_cropped_image(conf, TRN_CURATED/df_run.iloc[2,0])
df_test_bark, X_train_bark = convert_wav_to_cropped_image(df_bark, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)

df_test_run, X_train_run = convert_wav_to_cropped_image(df_run, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
# Just a quick dimension check here.

test_np_bark = np.vstack(X_train_bark)

test_np_run = np.vstack(X_train_run)

print(test_np_bark.shape)

print(test_np_run.shape)

end_time_data_processing = time.time()
df_train_curated = df_train_curated.sample(100) # taking a sample data set of 100 examples.

df_train_curated, X_train_curated = convert_wav_to_cropped_image(df_train_curated, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)

np_train_curated = np.vstack(X_train_curated)

X_train_curated = np_train_curated.reshape(-1, 128, 128, 3) # Reshaping the training dataset.

print(X_train_curated.shape)
X_train = X_train_curated

df_train = df_train_curated
# Getting the labels required for submission, there are eighty of them.

df_test = pd.read_csv('../input/sample_submission.csv')

label_columns = list( df_test.columns[1:] )

label_mapping = dict((label, index) for index, label in enumerate(label_columns))

#label_mapping

def split_and_label(rows_labels):

    row_labels_list = []

    for row in rows_labels:

        row_labels = row.split(',')

        labels_array = np.zeros((80))

        

        for label in row_labels:

            index = label_mapping[label]

            labels_array[index] = 1

        

        row_labels_list.append(labels_array)

    

    return row_labels_list
train_curated_labels = split_and_label(df_train['labels'])

for f in label_columns:

    df_train[f] = 0.0 # This adds all the labels as column names.



df_train[label_columns] = train_curated_labels
Y_train = np.vstack(train_curated_labels)

print(Y_train.shape)

print(X_train.shape)
df_test, X_test = convert_wav_to_cropped_image(df_test, source=TEST, img_dest=IMG_TEST)

np_test_small = np.vstack(X_test)

print(np_test_small.shape)

X_test = np_test_small.reshape(-1, 128, 128, 3)

print(X_test.shape)
print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))
import numpy as np

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.initializers import glorot_uniform

#from kt_utils import *



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
def Audio2DConvModel(input_shape, classes):

    """

    Implementation of the Basic Model.

    Arguments:

    input_shape -- shape of the images of the dataset

    Returns:

    model -- a Model() instance in Keras

    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (16, 16), strides = (1, 1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)

    #X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = MaxPooling2D(pool_size=(2, 2), name = 'maxpool_0')(X)

    X = Dropout(rate=0.1)(X)

# -------------------------------------------------------------------------------------

    X = ZeroPadding2D((3, 3))(X)

    X = Conv2D(32, (8, 8), strides = (1, 1), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2, 2), name = 'maxpool_1')(X)

    # ------------------------------------------------------------------------------------

    X = ZeroPadding2D((3, 3))(X)

    X = Conv2D(16, (4, 4), strides = (1, 1), name = 'conv2')(X)

    X = BatchNormalization(axis = 3, name = 'bn2')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='maxpool_2')(X)

    #------------------------------------------------------------

    X = ZeroPadding2D((3, 3))(X)

    X = Conv2D(16, (2, 2), strides = (1, 1), name = 'conv3')(X)

    X = BatchNormalization(axis = 3, name = 'bn32')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='maxpool_3')(X)

    #X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool_1')(X)

    # -------------------------------------------------------------------------------------    

    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc_softmax' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='Audio2DConvModel')

    return model
basic_conv_model = Audio2DConvModel(X_train.shape[1:], 80)

basic_conv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

basic_conv_model.fit(X_train/255, Y_train, epochs=1, batch_size=64)

basic_conv_model.summary()

y_hat = basic_conv_model.predict(X_test/255)

df_test[label_columns] = y_hat

print(df_test.head())