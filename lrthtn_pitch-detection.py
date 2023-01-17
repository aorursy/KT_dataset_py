import zipfile

import os

import json

import random

import itertools as it



import numpy as np

import pandas as pd



from scipy import signal

from scipy.io import wavfile



import sklearn as sk

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.svm import SVC



import matplotlib.pyplot as plt
def get_spectrogram(file_name, remove_begin=0, remove_end=0, filter_intensity=0):

    """

    Provide spectrogram associated to a WAV file

    To focus only on relevant information:

    - optionally removes begin and end of WAV file (ratio between 0 and 1)

    - optionally removes time buckets whose frequency intensities are all under a given threshold (absolute)

    """



    # Reading WAV file

    sample_rate, samples = wavfile.read(file_name)

    

    # Removing begin and end of samples since they may not be relevant

    length = len(samples)

    

    if remove_begin > 0:

        

        samples_to_remove = int(length * remove_begin)

        samples = samples[samples_to_remove:]

        

    if remove_end > 0:

        

        samples_to_remove = int(length * remove_end)

        samples = samples[:-samples_to_remove]

    

    # Computing spectogram

    frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate, nperseg=1024)



    # Getting max intensity for each time bucket

    max_intensity = np.amax(spectrogram, axis=0)



    # Filtering on max intensity

    selections = np.array(max_intensity > filter_intensity)



    return frequencies, times[selections], spectrogram[:, selections], max_intensity[selections], sample_rate, samples
def display_spectrogram(frequencies, times, spectrogram, sample_rate, samples):

    """

    Display spectrogram

    """

    

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        

    # Plotting frequencies for a given time

    axs[0].plot(frequencies, spectrogram[:, 50])

    axs[0].set_ylabel('Intensity')

    axs[0].set_xlabel('Frequency [Hz]')

    axs[0].set_title("Frequencies at arbitrary given time")



    # Plotting spectrogram (method 1)

    axs[1].pcolormesh(times, frequencies, 10*np.log10(spectrogram))

    axs[1].set_ylabel('Frequency [Hz]')

    axs[1].set_xlabel('Time [sec]')

    axs[1].set_title("Spectrogram (method 1)")



    # Plotting spectogram (method 2)

    axs[2].specgram(samples, Fs=sample_rate, NFFT=25, noverlap=5, detrend='mean', mode='psd')

    axs[2].set_ylabel('Frequency [Hz]')

    axs[2].set_xlabel('Time [sec]')

    axs[2].set_title("Spectrogram (method 2)")

    

    plt.show()
# Testing getting spectrogram

frequencies, times, spectrogram, _, sample_rate, samples = get_spectrogram(

    '../input/nsynth-test/nsynth-test/audio/keyboard_acoustic_004-031-050.wav',

    remove_begin=0.0,

    remove_end=0.0,

    filter_intensity=0)

display_spectrogram(frequencies, times, spectrogram, sample_rate, samples)



frequencies, times, spectrogram, _, sample_rate, samples = get_spectrogram(

    '../input/nsynth-test/nsynth-test/audio/vocal_synthetic_003-107-050.wav',

    remove_begin=0.0,

    remove_end=0.0,

    filter_intensity=0)

display_spectrogram(frequencies, times, spectrogram, sample_rate, samples)



frequencies, times, spectrogram, _, sample_rate, samples = get_spectrogram(

    '../input/nsynth-test/nsynth-test/audio/mallet_acoustic_062-025-100.wav',

    remove_begin=0.0,

    remove_end=0.0,

    filter_intensity=0)

display_spectrogram(frequencies, times, spectrogram, sample_rate, samples)
# Testing removing begin and end

_, times_raw, _, max_raw, _, _ = get_spectrogram('../input/nsynth-test/nsynth-test/audio/mallet_acoustic_062-025-100.wav',

                                                 remove_begin=0, remove_end=0, filter_intensity=0)



_, times_shortened, _, max_shortened, _, _ = get_spectrogram('../input/nsynth-test/nsynth-test/audio/mallet_acoustic_062-025-100.wav',

                                                             remove_begin=0.1, remove_end=0.1, filter_intensity=0)



fig, axs = plt.subplots(2, 1, figsize=(10, 10))



axs[0].plot(times_raw, max_raw, label='Raw')

axs[1].plot(times_shortened, max_shortened, label='Shortened')



axs[0].set_xlim(0, 6)

axs[0].set_ylim(0, 5e6)

axs[1].set_xlim(0, 6)

axs[1].set_ylim(0, 5e6)



axs[0].set_title('Raw')

axs[1].set_title('Shortened (10% begin and 10% end)')



fig.suptitle('Removing begin and end')



plt.show()
# Testing filtering on max intensity

_, times_raw, _, max_raw, _, _ = get_spectrogram('../input/nsynth-test/nsynth-test/audio/mallet_acoustic_062-025-100.wav',

                                                 remove_begin=0, remove_end=0, filter_intensity=0)



_, times_filtered, _, max_filtered, _, _ = get_spectrogram('../input/nsynth-test/nsynth-test/audio/mallet_acoustic_062-025-100.wav',

                                                           remove_begin=0, remove_end=0, filter_intensity=2e6)



fig=plt.figure(figsize=(10, 5))



plt.plot(times_raw, max_raw, label='Raw')

plt.plot(times_filtered, max_filtered, label='Filtered')



plt.legend(loc='upper right')



fig.suptitle('Filtering on max intensity')



plt.show()
# Reading JSON file containing metadata

json_metadata = open('../input/nsynth-test/nsynth-test/examples.json').read()

metadata = json.loads(json_metadata)
def prepare_data(file_name, samples, labels):

    

    # Getting spectrogram

    # Frequencies are frequency buckets

    # Times are time buckets

    # Intensities are intensities for each (frequency bucket, time bucket) couple

    # Max intensities are max intensities for each time bucket

    frequencies, times, intensities, max_intensities, _, _ = get_spectrogram(

        file_name,

        remove_begin=REMOVE_BEGIN,

        remove_end=REMOVE_END,

        filter_intensity=FILTER_INTENSITY)



    # Transposing spectrogram, to switch from frequencies x times to times x frequencies

    intensities = intensities.transpose()



    # Concatenating all time buckets in samples and labels sets

    # A time bucket is a list of 129 frequencies like [2.93450565e+01 5.87889600e+03 1.26233027e+04 2.72070879e+04 ... ]

    for time_bucket in intensities:



        samples.append(time_bucket)



        if labels is not None:

            

            # Pitch comes from metadata

            labels.append(value['pitch'])
REMOVE_BEGIN=0.0

REMOVE_END=0.0

FILTER_INTENSITY=1e6



samples = []

labels = []

    

# Looping on metadata

# Key would be something like "bass_synthetic_068-049-025"

# Value like {"qualities": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "pitch": 49, "note": 217499 ...

for key, value in metadata.items():



    prepare_data('../input/nsynth-test/nsynth-test/audio/' + key + '.wav', samples, labels)
# Scaling samples

scaler = StandardScaler().fit(samples)

samples = scaler.transform(samples)
# Shuffling and splitting, 1 split only at the beginning

shuffle_split = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)

shuffle_split.get_n_splits(samples)



for train_index, test_index in shuffle_split.split(samples):

    

    samples_train = np.asarray(samples)[train_index]

    samples_test = np.asarray(samples)[test_index]

    labels_train = np.asarray(labels)[train_index]

    labels_test = np.asarray(labels)[test_index]
# Computing train set distribution

distribution = [key for key, group in it.groupby(sorted(labels_train), key=lambda x:x)]



# Getting frequencies without corresponding sample

# 128 stands for number of different classes in labels

print(set(range(1, 129)) - set(distribution))
# Displaying train set distribution

# 128 stands for number of different classes in labels

fig=plt.figure(figsize=(10, 3))

plt.hist(labels_train, bins=128)

fig.suptitle('Labels distribution in train set')

plt.show()
for _ in range(1, 4):

    

    sample_index = random.randint(0, len(samples_train))

    fig=plt.figure(figsize=(10, 3))

    plt.plot(samples_train[sample_index])

    fig.suptitle(labels_train[sample_index])

    plt.show()
clf_logistic = LogisticRegression(random_state=0,

                                  solver='lbfgs',

                                  multi_class='multinomial',

                                  max_iter=500).fit(samples_train, labels_train)
clf_logistic.score(samples_train, labels_train)
clf_logistic.score(samples_test, labels_test)
#clf_linear_SVC = LinearSVC(C=1.0,

#                           class_weight=None,

#                           dual=False,

#                           fit_intercept=True,

#                           intercept_scaling=1,

#                           loss='squared_hinge',

#                           max_iter=200,

#                           multi_class='ovr', 

#                           penalty='l2', 

#                           random_state=0, 

#                           tol=1e-05, 

#                           verbose=0).fit(samples_train, labels_train)
#clf_linear_SVC.score(samples_train, labels_train)
#clf_linear_SVC.score(samples_test, labels_test)
#clf_SVC = SVC(kernel='rbf',

#              gamma=0.7,

#              C=1.0).fit(samples_train, labels_train)
#clf_SVC.score(samples_train, labels_train)
#clf_SVC.score(samples_test, labels_test)
clf_neural = MLPClassifier(solver='lbfgs',

                           alpha=1e-5,

                           hidden_layer_sizes=(16),

                           random_state=1).fit(samples_train, labels_train)
clf_neural.score(samples_train, labels_train)
clf_neural.score(samples_test, labels_test)
train_sizes, train_scores, valid_scores = learning_curve(

    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16), random_state=1),

    samples,

    labels,

    train_sizes=[100, 1000, 5000, 20000, 50000],

    cv=3,

    n_jobs=3,

    shuffle=True)



# Plotting learning curve for both train scores and test scores

plt.plot(train_sizes, train_scores[:, 0], c='b', label='Train - Fold 0')

plt.plot(train_sizes, train_scores[:, 1], c='g', label='Train - Fold 1')

plt.plot(train_sizes, train_scores[:, 2], c='r', label='Train - Fold 2')

plt.plot(train_sizes, valid_scores[:, 0], c='c', label='Test - Fold 0')

plt.plot(train_sizes, valid_scores[:, 1], c='m', label='Test - Fold 1')

plt.plot(train_sizes, valid_scores[:, 2], c='y', label='Test - Fold 2')

plt.legend(loc='lower right');

plt.title = 'Learning curves'

plt.grid()

plt.show()
# Getting predictions

predictions = clf_neural.predict(samples_test)



# Getting indexes where predictions differ from ground truth

error_indexes = (predictions !=  labels_test)



# Computing error for each of these indexes

delta = predictions[error_indexes] - labels_test[error_indexes]

delta = [abs(number) for number in delta]



# Computing average of these errors

# 128 stands for number of different classes in labels

print(sum(delta) / len(delta) / 128)
# Displaying one of these errors

plt.plot(samples_test[error_indexes][0])

plt.show()
samples_prediction = []



# Preprocessing

prepare_data('../input/nsynth-test/nsynth-test/audio/guitar_acoustic_010-056-050.wav', samples_prediction, None)

samples_prediction_scaled = scaler.transform(samples_prediction)

samples_prediction_scaled = np.asarray(samples_prediction_scaled)



# Prediction

clf_neural.predict(samples_prediction_scaled)



# Comparison with ground truth is easy since it is embedded in filename xxxxx_xxx-056-xxx.wav