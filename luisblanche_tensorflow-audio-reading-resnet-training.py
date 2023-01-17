import os

import librosa

import librosa.display

import pathlib



import numpy as np

import pandas as pd

import tensorflow as tf

from pydub import AudioSegment



import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder



AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
N_CLASSES = 264 # Allows to reduce the number of classes to train (max = 264)

SAMPLE_RATE = 32000 # Audio sample rate

MAX_DURATION = 5 # Clip duration in seconds 

FFT_SIZE = 1024 # Fourier Transform size 

HOP_SIZE = 512 # Number of samples between each successive FFT window

N_MEL_BINS = 128 

N_SPECTROGRAM_BINS = (FFT_SIZE // 2) + 1

F_MIN = 20# Min frequency cutoff

F_MAX = SAMPLE_RATE / 2  # Max Frequency cutoff

BATCH_SIZE = 64  # Training Batch size
train = pd.read_csv("../input/birdsong-recognition/train.csv", parse_dates=['date'])

train.head()
input_paths = {'a':'../input/birdsong-resampled-train-audio-00',

               'b': '../input/birdsong-resampled-train-audio-00',

               'c': '../input/birdsong-resampled-train-audio-01',

               'e': '../input/birdsong-resampled-train-audio-01',

               'f': '../input/birdsong-resampled-train-audio-01',

               'g': '../input/birdsong-resampled-train-audio-02',

               'h': '../input/birdsong-resampled-train-audio-02',

               'i': '../input/birdsong-resampled-train-audio-02',

               'j': '../input/birdsong-resampled-train-audio-02',

               'k': '../input/birdsong-resampled-train-audio-02',

               'l': '../input/birdsong-resampled-train-audio-02',

               'm': '../input/birdsong-resampled-train-audio-02',

               'n': '../input/birdsong-resampled-train-audio-03',

               'o': '../input/birdsong-resampled-train-audio-03',

               'p': '../input/birdsong-resampled-train-audio-03',

               'q': '../input/birdsong-resampled-train-audio-03',

               'r': '../input/birdsong-resampled-train-audio-03',

               's': '../input/birdsong-resampled-train-audio-04',

               't': '../input/birdsong-resampled-train-audio-04',

               'u': '../input/birdsong-resampled-train-audio-04',

               'v': '../input/birdsong-resampled-train-audio-04',

               'w': '../input/birdsong-resampled-train-audio-04',

               'x': '../input/birdsong-resampled-train-audio-04',

               'y': '../input/birdsong-resampled-train-audio-04'          

        }









train['filepath'] = train["ebird_code"].str[0].map(input_paths) + '/' + train["ebird_code"] + '/' + train["filename"]

train['filepath'] = train['filepath'].str.replace('.mp3', '.wav')

train['ebird_code'].value_counts()[:N_CLASSES]
train = train.dropna(subset=['filepath'])
le = LabelEncoder()

train['label'] = le.fit_transform(train['ebird_code'])
import pandas as pd

import tensorflow as tf



AUTOTUNE = tf.data.experimental.AUTOTUNE





def get_dataset(df, label_column, filepath_column):

    file_path_ds = tf.data.Dataset.from_tensor_slices(df[filepath_column].astype(bytes))

    label_ds = tf.data.Dataset.from_tensor_slices(df[label_column])

    return tf.data.Dataset.zip((file_path_ds, label_ds))





def load_audio(file_path, label):

    audio = tf.io.read_file(file_path)

    audio, sample_rate = tf.audio.decode_wav(audio,

                                             desired_channels=1, 

                                             desired_samples = SAMPLE_RATE * 60  # take first 60 seconds (no offset possible ..)

                                           )

    #audio = tf.transpose(audio)

    audio = tf.image.random_crop(audio, size=[SAMPLE_RATE * MAX_DURATION, 1]) # Random crop to 5 seconds

    return audio, label







def prepare_for_training(ds, shuffle_buffer_size=512, batch_size=64):

    # Randomly shuffle (file_path, label) dataset

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Load and decode audio from file paths

    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)

    # Repeat dataset forever

    ds = ds.repeat()

    # Prepare batches

    ds = ds.batch(batch_size)

    # Prefetch

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
load_audio(train.loc[0, 'filepath'], 1)

from sklearn.model_selection import train_test_split

train, val = train_test_split(train, stratify=train['label'], test_size=0.1)
train_ds = get_dataset(train, 'label', 'filepath')

val_ds = get_dataset(val, 'label', 'filepath')
class LogMelSpectrogram(tf.keras.layers.Layer):

    """Compute log-magnitude mel-scaled spectrograms."""



    def __init__(self, sample_rate, fft_size, hop_size, n_mels,

                 f_min=0.0, f_max=None, **kwargs):

        super(LogMelSpectrogram, self).__init__(**kwargs)

        self.sample_rate = sample_rate

        self.fft_size = fft_size

        self.hop_size = hop_size

        self.n_mels = n_mels

        self.f_min = f_min

        self.f_max = f_max if f_max else sample_rate / 2

        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(

            num_mel_bins=self.n_mels,

            num_spectrogram_bins=fft_size // 2 + 1,

            sample_rate=self.sample_rate,

            lower_edge_hertz=self.f_min,

            upper_edge_hertz=self.f_max)



    def build(self, input_shape):

        self.non_trainable_weights.append(self.mel_filterbank)

        super(LogMelSpectrogram, self).build(input_shape)



    def call(self, waveforms):

        """Forward pass.



        Parameters

        ----------

        waveforms : tf.Tensor, shape = (None, n_samples)

            A Batch of mono waveforms.



        Returns

        -------

        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)

            The corresponding batch of log-mel-spectrograms

        """

        def _tf_log10(x):

            numerator = tf.math.log(x)

            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))

            return numerator / denominator



        def power_to_db(magnitude, amin=1e-16, top_db=80.0):

            """

            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html

            """

            ref_value = tf.reduce_max(magnitude)

            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))

            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))

            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)



            return log_spec



        spectrograms = tf.signal.stft(waveforms,

                                      frame_length=self.fft_size,

                                      frame_step=self.hop_size,

                                      pad_end=False)



        magnitude_spectrograms = tf.abs(spectrograms)



        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),

                                     self.mel_filterbank)



        log_mel_spectrograms = power_to_db(mel_spectrograms)



        # add channel dimension

        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms



    def get_config(self):

        config = {

            'fft_size': self.fft_size,

            'hop_size': self.hop_size,

            'n_mels': self.n_mels,

            'sample_rate': self.sample_rate,

            'f_min': self.f_min,

            'f_max': self.f_max,

        }

        config.update(super(LogMelSpectrogram, self).get_config())



        return config
#resnet = tf.keras.applications.ResNet50(

#    include_top=False, weights='imagenet', input_shape=(311, 128, 3),

#    pooling='avg'

#)



import tensorflow_hub as hub



feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,

                                         input_shape=(311, 128, 3))



feature_extractor_layer.trainable = False
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,

                                     Dropout, Flatten, Input, MaxPool2D)

from tensorflow.keras.models import Model



def ConvModel(n_classes, sample_rate=SAMPLE_RATE, duration=MAX_DURATION,

              fft_size=FFT_SIZE, hop_size=HOP_SIZE, n_mels=N_MEL_BINS, fmin=F_MIN, fmax=F_MAX):

    n_samples = sample_rate * duration

    input_shape = (n_samples,)



    x = Input(shape=input_shape, name='input', dtype='float32')    

    y = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels, fmin, fmax)(x)

    y = BatchNormalization(axis=2)(y)





    y = Conv2D(3, (3,3), padding='same')(y)  

    y = BatchNormalization()(y)



    y = feature_extractor_layer(y, training=False)



    y = Dense(1024, activation='relu')(y)

    y = Dropout(0.1)(y)

    y = Dense(1024, activation='relu')(y)

    y = Dropout(0.1)(y)

    

    y = Dense(n_classes, activation='softmax')(y)



    return Model(inputs=x, outputs=y)
from tensorflow.keras.optimizers import SGD, schedules



n_classes = train['label'].max() + 1

model = ConvModel(n_classes)



lr_schedule = schedules.ExponentialDecay(

    initial_learning_rate=0.05, decay_steps=1000, decay_rate=0.96, staircase=False

)

sgd = SGD(learning_rate=lr_schedule, momentum=0.85)

model.compile(optimizer=sgd,

              loss='sparse_categorical_crossentropy', 

              metrics=['sparse_categorical_accuracy'])



model.summary()

training_ds = prepare_for_training(train_ds)

valid_ds = prepare_for_training(val_ds)



steps_per_epoch = len(train)//BATCH_SIZE

steps_per_epoch
checkpoint_filepath = '/checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_filepath,

    save_weights_only=True,

    monitor='val_acc',

    mode='max',

    save_best_only=True)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)



model.fit(training_ds, 

          epochs=50, 

          steps_per_epoch=steps_per_epoch, 

          validation_data=valid_ds, 

          validation_steps=2, 

         callbacks=[model_checkpoint_callback, early_stop])
model.save("resnet_model.h5")
for pred, label in valid_ds.take(2):

    pred = pred.numpy()

    label = label.numpy()
pred.shape
label
label
predict = model.predict(pred)
predict.shape
[pred.argmax() for pred in predict]
np.where(predict[0]>0.008)
np.where(predict[1]>0.008)