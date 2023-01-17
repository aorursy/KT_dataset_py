import os

from os.path import isdir, join

from pathlib import Path

import pandas as pd

import time



# Math

import numpy as np

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

import librosa



from sklearn.decomposition import PCA



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import IPython.display as ipd

import librosa.display



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import pandas as pd



%matplotlib inline

t1 = time.time()
##

## Read necessary files and folders 

train_audio_path = '../input/freesound-audio-tagging-2019/train_curated/'

train_files = os.listdir(train_audio_path)

train_annot = pd.read_csv('../input/freesound-audio-tagging-2019/train_curated.csv')

test_audio_path = '../input/freesound-audio-tagging-2019/test/'

test_files = np.sort(os.listdir(test_audio_path))

print (test_files[:5])

##

## I 've created the weights in previous run. Just use them

USE_WEIGHTS = False

len(train_files), len(test_files)

##

## Here we calculate unique labels (80) and create the necessary data structures

## for binary encoding the multi-labeled input

##

## label_dict : dictionary with classes as keys and counts per classes as values

##              {'Bark': 74, 'Raindrop': 74, 'Finger_snapping': 74, 'Run': 74, 'Whispering': 74, .... }

## classes: 80 sound classes 

##              ['Bark', 'Raindrop', 'Finger_snapping', 'Run', 'Whispering', ...]

## all_labels_set (size: 4970): List of sets. The same size as training sounds. Each set correspond to the classes of the i-th sound

##              [{'Bark'},  {'Raindrop'},  {'Finger_snapping'},  {'Run'},  {'Finger_snapping'},  {'Whispering'},  {'Acoustic_guitar', 'Strum'},  ...]

## first_labels_set (size: 4970) : List containing only first class for each training pattern..  to be used as approximation stratification 

##              ['Bark', 'Raindrop', 'Finger_snapping', 'Run', 'Finger_snapping', 'Whispering', 'Acoustic_guitar', ...]



##

## L = 1 * SAMPLE_RATE -> 1 second

## L = 2 * SAMPLE_RATE -> 2 seconds ....



SAMPLE_RATE  = 44100

L = int( 1.5 * SAMPLE_RATE )



def create_unique_labels(all_labels):

    label_dict = {}

    all_labels_set = []

    first_labels_set = []

    for labs in all_labels:

        lab = labs.split(',')

        for l in lab:

            if l in label_dict:

                label_dict[l] = label_dict[l]  + 1

            else:

                label_dict[l]= 0



        all_labels_set.append(set(lab))

        first_labels_set.append(lab[0])

    classes = list(label_dict.keys())

    

    return label_dict, classes, all_labels_set, first_labels_set
label_dict, classes, all_labels_set, first_labels_set = create_unique_labels(train_annot.labels)

files = train_annot.fname

print (len(files), len(train_files))
##

## Y_split are the binary labels used for stratification

## Y is the target

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

binarize = MultiLabelBinarizer(classes=classes)

encode = LabelEncoder()

Y_split = encode.fit_transform(first_labels_set)

Y = binarize.fit_transform(all_labels_set)
##

## Read all training files and keep them in memory

from tqdm import tqdm_notebook

X_raw = []

for f in tqdm_notebook(files):

    sample_rate, sample_ = wavfile.read(str(train_audio_path) + f)

    X_raw.append(sample_)
##

## Nice helper functions for padding, random sampling L samples



def pad_audio(samples):

    if len(samples) >= L: return samples

    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))



# 150000 , 44100 ->  [0, .,......., (150000-44100)]

def chop_audio(samples):

    beg = np.random.randint(0, len(samples) - L)

    return samples[beg: beg + L]

        

        

def log_specgram(audio, 

                 sample_rate, 

                 window_size=20,

                 step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))

    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, times, spec = signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return freqs, times, np.log(spec.astype(np.float32) + eps)
##

## DataGenerator based on keras.utils.Sequence. The nice thing about it is the random part selection that works like augmentation.

## TestDataGenerator is bad software engineering from my part... Essentially the same generator used only for inference...



import numpy as np

import keras



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(256,256,1), n_channels=1,

                 n_classes=80, shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.ceil(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        max_index = min((index+1)*self.batch_size, len(self.list_IDs))

        indexes = self.indexes[index*self.batch_size:max_index]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = []# np.empty((self.batch_size, *self.dim, self.n_channels))

        y = []# np.empty((self.batch_size), dtype=int)

        t1 = time.time()

        #print (list_IDs_temp)



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            #print (i, ID)

            # Store samplw

            xx = X_raw[ID].copy()

    

            xx = pad_audio(xx)

            if len(xx) > L:

                xx = chop_audio(xx)

            _, _, specgram = log_specgram(xx, sample_rate=SAMPLE_RATE,  window_size=10, step_size=5)

            X.append(specgram)



            # Store class

            y.append(self.labels[ID, :])

            

        t2 = time.time()

        #print (t2-t1)

        y = np.array(y, dtype='float32')

        X = np.expand_dims(np.array(X), -1)

        return X, y



class TestDataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, test_files, test_base_path, batch_size=32, dim=(256,256,1), n_channels=1,

                 n_classes=80):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.test_base_path = test_base_path

        self.test_files = test_files

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.ceil(len(self.test_files) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        max_index = min((index+1)*self.batch_size, len(self.test_files))

        indexes = self.indexes[index*self.batch_size:max_index]



        # Find list of IDs

        list_IDs_temp = [self.test_files[k] for k in indexes]



        # Generate data

        X = self.__data_generation(list_IDs_temp)



        return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.test_files))





    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = []# np.empty((self.batch_size, *self.dim, self.n_channels))



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store samplw

            sample_rate, xx = wavfile.read(str(self.test_base_path) + ID)

            xx = pad_audio(xx)

            if len(xx) > L:

                xx = chop_audio(xx)

            _, _, specgram = log_specgram(xx, sample_rate=SAMPLE_RATE,  window_size=10, step_size=5)

            X.append(specgram)



        X = np.expand_dims(np.array(X), -1)

        return X
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,

                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, Dropout)

from keras.utils import Sequence, to_categorical

from keras.optimizers import Adam

from keras.losses import binary_crossentropy

from keras.models import Model

from keras import backend as K



def get_2d_conv_model(input_shape= (221, 298, 1), n_classes=80, learning_rate=0.001):

    

    nclass = n_classes

    

    inp = Input(shape=(input_shape[0],input_shape[1],1))

    x = Convolution2D(96, (4,10), padding="same")(inp)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D()(x)

    

    x = Convolution2D(64, (4,10), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D()(x)

    

    x = Convolution2D(48, (4,10), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D()(x)

    

    x = Convolution2D(32, (4,10), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D()(x)

    x = Flatten()(x)

    

    x = Dropout(0.5)(x)

    x = Dense(80)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    

    x = Dense(80)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    

    out = Dense(nclass, activation='softmax')(x)



    model = Model(inputs=inp, outputs=out)

    opt = Adam(learning_rate)



    model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['categorical_accuracy'])

    return model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

modelname = 'custom-v1-'

test_gen = TestDataGenerator(test_files, test_audio_path, batch_size=128)
from sklearn.model_selection import StratifiedKFold

oof_y = np.zeros_like(Y, dtype='float32')

test_Y = np.zeros((len(test_files), 80), dtype='float32')



#90%/10% Split

kfold = StratifiedKFold(10, True, 6996 )

ifold = 0

for train_index, valid_index in kfold.split(X_raw, Y_split):

    if ifold<=1:

        print("--------------------------------------------")



        checkpoint = ModelCheckpoint(modelname + str(ifold) + '.hdf5', monitor='val_categorical_accuracy', verbose=1, 

                                     save_best_only=True, save_weights_only=True, mode='auto', period=1)

        early = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.001, patience=12, verbose=1, mode='auto', restore_best_weights=True)



        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=6, verbose=1)



        model = get_2d_conv_model()

        train_gen = DataGenerator(train_index, Y, batch_size=32, n_classes=80, shuffle=True)

        valid_gen = DataGenerator(valid_index, Y, batch_size=128, n_classes=80, shuffle=False)



#         if USE_WEIGHTS == True:

#             print ('Loading from ', '../input/custom-cnn-1-sec/' + modelname + str(ifold) + '.hdf5')

#             model.load_weights('../input/custom-cnn-1-sec/' + modelname + str(ifold) + '.hdf5')

#         else:

        model.fit_generator(train_gen, epochs=60, verbose=0, callbacks=[checkpoint, early, reduce_lr], validation_data=valid_gen)



        res = model.predict_generator(valid_gen, verbose=0)

        #for bag in range(9):

        #    res += model.predict_generator(valid_gen, verbose=0)



        res_Y = model.predict_generator(test_gen, verbose=0)

        for bag in range(9):

            res_Y += model.predict_generator(test_gen, verbose=0)



        oof_y[valid_index, ] = res

        test_Y = test_Y + res_Y / 10.

    ifold = ifold + 1

    

test_Y = test_Y / 5

import numpy as np

import sklearn.metrics

# Core calculation of label precisions for one test sample.



def _one_sample_positive_class_precisions(scores, truth):

  """Calculate precisions for each true class for a single sample.

  

  Args:

    scores: np.array of (num_classes,) giving the individual classifier scores.

    truth: np.array of (num_classes,) bools indicating which classes are true.



  Returns:

    pos_class_indices: np.array of indices of the true classes for this sample.

    pos_class_precisions: np.array of precisions corresponding to each of those

      classes.

  """

  num_classes = scores.shape[0]

  pos_class_indices = np.flatnonzero(truth > 0)

  # Only calculate precisions if there are some true classes.

  if not len(pos_class_indices):

    return pos_class_indices, np.zeros(0)

  # Retrieval list of classes for this sample. 

  retrieved_classes = np.argsort(scores)[::-1]

  # class_rankings[top_scoring_class_index] == 0 etc.

  class_rankings = np.zeros(num_classes, dtype=np.int)

  class_rankings[retrieved_classes] = range(num_classes)

  # Which of these is a true label?

  retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

  retrieved_class_true[class_rankings[pos_class_indices]] = True

  # Num hits for every truncated retrieval list.

  retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

  # Precision of retrieval list truncated at each hit, in order of pos_labels.

  precision_at_hits = (

      retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 

      (1 + class_rankings[pos_class_indices].astype(np.float)))

  return pos_class_indices, precision_at_hits



# All-in-one calculation of per-class lwlrap.



def calculate_per_class_lwlrap(truth, scores):

  """Calculate label-weighted label-ranking average precision.

  

  Arguments:

    truth: np.array of (num_samples, num_classes) giving boolean ground-truth

      of presence of that class in that sample.

    scores: np.array of (num_samples, num_classes) giving the classifier-under-

      test's real-valued score for each class for each sample.

  

  Returns:

    per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each 

      class.

    weight_per_class: np.array of (num_classes,) giving the prior of each 

      class within the truth labels.  Then the overall unbalanced lwlrap is 

      simply np.sum(per_class_lwlrap * weight_per_class)

  """

  assert truth.shape == scores.shape

  num_samples, num_classes = scores.shape

  # Space to store a distinct precision value for each class on each sample.

  # Only the classes that are true for each sample will be filled in.

  precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

  for sample_num in range(num_samples):

    pos_class_indices, precision_at_hits = (

      _one_sample_positive_class_precisions(scores[sample_num, :], 

                                            truth[sample_num, :]))

    precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

        precision_at_hits)

  labels_per_class = np.sum(truth > 0, axis=0)

  weight_per_class = labels_per_class / float(np.sum(labels_per_class))

  # Form average of each column, i.e. all the precisions assigned to labels in

  # a particular class.

  per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) / 

                      np.maximum(1, labels_per_class))

  # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

  #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

  #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

  #                = np.sum(per_class_lwlrap * weight_per_class)

  return per_class_lwlrap, weight_per_class



# Calculate the overall lwlrap using sklearn.metrics function.



def calculate_overall_lwlrap_sklearn(truth, scores):

  """Calculate the overall lwlrap using sklearn.metrics.lrap."""

  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

  sample_weight = np.sum(truth > 0, axis=1)

  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(

      truth[nonzero_weight_sample_indices, :] > 0, 

      scores[nonzero_weight_sample_indices, :], 

      sample_weight=sample_weight[nonzero_weight_sample_indices])

  return overall_lwlrap





# Accumulator object version.



class lwlrap_accumulator(object):

  """Accumulate batches of test samples into per-class and overall lwlrap."""  



  def __init__(self):

    self.num_classes = 0

    self.total_num_samples = 0

  

  def accumulate_samples(self, batch_truth, batch_scores):

    """Cumulate a new batch of samples into the metric.

    

    Args:

      truth: np.array of (num_samples, num_classes) giving boolean

        ground-truth of presence of that class in that sample for this batch.

      scores: np.array of (num_samples, num_classes) giving the 

        classifier-under-test's real-valued score for each class for each

        sample.

    """

    assert batch_scores.shape == batch_truth.shape

    num_samples, num_classes = batch_truth.shape

    if not self.num_classes:

      self.num_classes = num_classes

      self._per_class_cumulative_precision = np.zeros(self.num_classes)

      self._per_class_cumulative_count = np.zeros(self.num_classes, 

                                                  dtype=np.int)

    assert num_classes == self.num_classes

    for truth, scores in zip(batch_truth, batch_scores):

      pos_class_indices, precision_at_hits = (

        _one_sample_positive_class_precisions(scores, truth))

      self._per_class_cumulative_precision[pos_class_indices] += (

        precision_at_hits)

      self._per_class_cumulative_count[pos_class_indices] += 1

    self.total_num_samples += num_samples



  def per_class_lwlrap(self):

    """Return a vector of the per-class lwlraps for the accumulated samples."""

    return (self._per_class_cumulative_precision / 

            np.maximum(1, self._per_class_cumulative_count))



  def per_class_weight(self):

    """Return a normalized weight vector for the contributions of each class."""

    return (self._per_class_cumulative_count / 

            float(np.sum(self._per_class_cumulative_count)))



  def overall_lwlrap(self):

    """Return the scalar overall lwlrap for cumulated samples."""

    return np.sum(self.per_class_lwlrap() * self.per_class_weight())

truth = Y

scores = oof_y

print("lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores))
sort_idx = np.argsort(classes).astype(int)

sample_sub = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')

test_Y_sort = test_Y[:, sort_idx]

sample_sub.iloc[:, 1:] =  test_Y_sort

sample_sub.to_csv('submission.csv', index=False)



t2 = time.time()

print ('Total time: ', (t2-t1))
sample_sub.head()