%matplotlib inline

import numpy as np

import pandas as pd

from scipy.io import loadmat

import glob, re, math

import matplotlib.pyplot as plt

import seaborn as sns
def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



# Generate the feature specified by 'func' with each matching file split into 'num_splits' parts

def generate_feature(file_pattern, num_splits, func):

    files = sorted(glob.glob(file_pattern), key=natural_key)

    n_files = len(files)

    feature = np.zeros((n_files*num_splits,16))

    for i in range(n_files):

        path = files[i]

        try:

            mat = loadmat(path)

            data = mat['dataStruct']['data'][0, 0]

            split_length = data.shape[0]/num_splits

            for s in range(num_splits):

                split_start = split_length*s

                split_end = split_start+split_length

                for c in range(16):

                    channel_data = data[split_start:split_end,c]

                    zero_fraction = float(channel_data.size - np.count_nonzero(channel_data))/channel_data.size

                    # Exclude sections with more than 10% dropout

                    if zero_fraction > 0.1:

                        feature[i*num_splits+s,c] = float('nan')

                    else:

                        feature[i*num_splits+s,c] = func(channel_data)

        except:

            for s in range(num_splits):

                for c in range(16):

                    feature[i*num_splits+s,c] = float('nan')

    return feature



# Simple log energy feature

def log_energy(data):

    return math.log(np.std(data))
train1_negative_log_energy = generate_feature('../input/train_1/*0.mat', 60, log_energy)

test1_log_energy = generate_feature('../input/test_1/*.mat', 60, log_energy)



train2_negative_log_energy = generate_feature('../input/train_2/*0.mat', 60, log_energy)

test2_log_energy = generate_feature('../input/test_2/*.mat', 60, log_energy)
sns.distplot(train1_negative_log_energy[:,13][~np.isnan(train1_negative_log_energy[:,13])], axlabel='Log energy (Channel 13, Patient 1)')

sns.distplot(test1_log_energy[:,13][~np.isnan(test1_log_energy[:,13])], label='Test')
sns.distplot(train1_negative_log_energy[:,5][~np.isnan(train1_negative_log_energy[:,5])], axlabel='Log energy (Channel 5, Patient 1)')

sns.distplot(test1_log_energy[:,5][~np.isnan(test1_log_energy[:,5])])
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

ax1.plot(train1_negative_log_energy[:,13], '.', ms=1)

ax1.set_xlabel('Train')

ax2.plot(test1_log_energy[:,13], '.', ms=1)

ax2.set_xlabel('Test')

ax1.set_title('Log energy, Patient 1, Channel 13')

plt.show()
sns.distplot(train2_negative_log_energy[:,3][~np.isnan(train2_negative_log_energy[:,3])], axlabel='Log energy (Channel 3, Patient 2)')

sns.distplot(test2_log_energy[:,3][~np.isnan(test2_log_energy[:,3])])
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

ax1.plot(train2_negative_log_energy[:,3], '.', ms=1)

ax1.set_xlabel('Train')

ax2.plot(test2_log_energy[:,3], '.', ms=1)

ax2.set_xlabel('Test')

ax1.set_title('Log energy, Patient 2, Channel 3')

plt.show()
sns.distplot(train2_negative_log_energy[:,9][~np.isnan(train2_negative_log_energy[:,9])], axlabel='Log energy (Channel 9, Patient 2)')

sns.distplot(test2_log_energy[:,9][~np.isnan(test2_log_energy[:,9])])
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

ax1.plot(train2_negative_log_energy[:,9], '.', ms=1)

ax1.set_xlabel('Train')

ax2.plot(test2_log_energy[:,9], '.', ms=1)

ax2.set_xlabel('Test')

ax1.set_title('Log energy, Patient 2, Channel 9')

plt.show()