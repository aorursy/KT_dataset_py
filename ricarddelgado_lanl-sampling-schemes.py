import gc

import os

import time

import random

import datetime

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GridSearchCV, cross_val_score
%matplotlib inline

pd.options.display.precision = 15

random.seed(6) #totally random seed (got from a dice)
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

fs = 4000000 #sampling frequency of the sensor signal
print(f'Train: rows:{train.shape[0]} cols:{train.shape[1]}')
train_acoustic_data_small = train['acoustic_data'].values[::50]

train_time_to_failure_small = train['time_to_failure'].values[::50]



fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure (sampled)')

plt.plot(train_acoustic_data_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_time_to_failure_small, color='g')

ax2.set_ylabel('time_to_failure', color='g')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)



del train_acoustic_data_small

del train_time_to_failure_small
plt.figure()

plt.hist(train['time_to_failure'].values[::50], bins='auto', density=True)  # arguments are passed to np.histogram

plt.title('Histogram of time_to_failure')

plt.xlabel('time_to_failure')

plt.ylabel('count')

plt.show()
# Create a training file with simple derived features

segment_size = 150000
def generate_segment_start_ids(sampling_method):

    """ Generates the indeces where the segments for the training data start """

    if sampling_method == 'uniform':

        # With this approach we obtain 4194 segments

        num_segments_training = int(np.floor(train.shape[0] / segment_size))

        segment_start_ids = [i * segment_size for i in range(num_segments_training)]

    elif sampling_method == 'uniform_no_jump':

        # With this approach we obtain 4178 segments (99.5% of 'uniform')

        already_sampled = np.full(train.shape[0], False)

        num_segments_training = int(np.floor(train.shape[0] / segment_size))

        time_to_failure_jumps = np.diff(train['time_to_failure'].values)

        num_good_segments_found = 0

        segment_start_ids = []

        for i in range(num_segments_training):

            idx = i * segment_size

            # Detect if there is a discontinuity on the time_to_failure signal within the segment

            max_jump = np.max(time_to_failure_jumps[idx:idx + segment_size])

            if max_jump < 5:

                segment_start_ids.append(idx)

                num_good_segments_found += 1

            else:

                print(f'Rejected candidate segment since max_jump={max_jump}')

        segment_start_ids.sort()

    elif sampling_method == 'random_no_jump':

        # With this approach we obtain 4194 segments

        num_segments_training = int(np.floor(train.shape[0] / segment_size)) #arbitrary choice

        time_to_failure_jumps = np.diff(train['time_to_failure'].values)

        num_good_segments_found = 0

        segment_start_ids = []

        while num_segments_training != num_good_segments_found:

            # Generate a random sampling position

            idx = random.randint(0, train.shape[0] - segment_size - 1)

            # Detect if there is a discontinuity on the time_to_failure signal within the segment

            max_jump = np.max(time_to_failure_jumps[idx:idx + segment_size])

            if max_jump < 5:

                segment_start_ids.append(idx)

                num_good_segments_found += 1

            else:

                print(f'Rejected candidate segment since max_jump={max_jump}')

        segment_start_ids.sort()

    else:

        raise NameError('Method does not exist')

    return segment_start_ids
print(f'Generating uniformly sampled training set')

segment_start_ids_uniform = generate_segment_start_ids('uniform')



print(f'Generating uniformly sampled training set excluding discontinuities in time_to_failure.')

segment_start_ids_uniform_no_jump = generate_segment_start_ids('uniform_no_jump')



print(f'Generating randomly sampled training set excluding discontinuities in time_to_failure.')

print(f'This method may yield overlaping segments')

segment_start_ids_random_no_jump = generate_segment_start_ids('random_no_jump')





y_tr_samples_uniform = train['time_to_failure'].values[np.array(

    segment_start_ids_uniform) + segment_size - 1]

y_tr_samples_uniform_no_jump = train['time_to_failure'].values[

    np.array(segment_start_ids_uniform_no_jump) + segment_size - 1]

y_tr_samples_random_no_jump = train['time_to_failure'].values[

    np.array(segment_start_ids_random_no_jump) + segment_size - 1]



plt.subplots(figsize=(16, 5))

plt.subplot(1, 3, 1)

plt.hist(y_tr_samples_uniform, bins='auto', alpha=0.5, density=True)

plt.hist(train['time_to_failure'].values[::50], bins='auto', alpha=0.5, density=True)

plt.title('With discontinuities (contiguous)')

plt.legend(['Sampled', 'All data'])



plt.subplot(1, 3, 2)

plt.hist(y_tr_samples_uniform_no_jump, bins='auto', alpha=0.5, density=True)

plt.hist(train['time_to_failure'].values[::50], bins='auto', alpha=0.5, density=True)

plt.title('Discarding discontinuities (contiguous)')

plt.legend(['Sampled', 'All data'])



plt.subplot(1, 3, 3)

plt.hist(y_tr_samples_random_no_jump, bins='auto', alpha=0.5, density=True)

plt.title('Discarding discontinuities (rand)')

plt.hist(train['time_to_failure'].values[::50], bins='auto', alpha=0.5, density=True)

plt.legend(['Sampled', 'All data'])

plt.show()
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
n_fold = folds.get_n_splits()



for fold_n, (train_index, valid_index) in enumerate(folds.split(y_tr_samples_uniform_no_jump)):

    print('Fold', fold_n, 'started at', time.ctime())

    y_train, y_valid = y_tr_samples_uniform_no_jump[train_index], y_tr_samples_uniform_no_jump[valid_index]



    plt.figure()

    plt.hist(y_train, bins='auto', alpha=0.5, density=True)

    plt.hist(y_valid, bins='auto', alpha=0.5, density=True)

    plt.title(f"Histogram with fold {fold_n}")

    plt.legend([f'Train (median={np.median(y_train):.4f})', f'Validation (median={np.median(y_valid):.4f})'])

    plt.show()