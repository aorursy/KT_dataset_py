import warnings

import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix , log_loss



warnings.simplefilter(action = 'ignore')

sns.set_style('dark')
dtypes = {"crew": "int8",

          "experiment": "category",

          "time": "float32",

          "seat": "int8",

          "eeg_fp1": "float32",

          "eeg_f7": "float32",

          "eeg_f8": "float32",

          "eeg_t4": "float32",

          "eeg_t6": "float32",

          "eeg_t5": "float32",

          "eeg_t3": "float32",

          "eeg_fp2": "float32",

          "eeg_o1": "float32",

          "eeg_p3": "float32",

          "eeg_pz": "float32",

          "eeg_f3": "float32",

          "eeg_fz": "float32",

          "eeg_f4": "float32",

          "eeg_c4": "float32",

          "eeg_p4": "float32",

          "eeg_poz": "float32",

          "eeg_c3": "float32",

          "eeg_cz": "float32",

          "eeg_o2": "float32",

          "ecg": "float32",

          "r": "float32",

          "gsr": "float32",

          "event": "category",

         }
train_df = pd.read_csv('../input/reducing-commercial-aviation-fatalities/train.csv',dtype=dtypes)

test_df = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv',dtype=dtypes)
train_df.info()
train_df.sample(10)
test_df.head()
plt.figure(figsize = (15,10))

sns.countplot(train_df['event'])

plt.xlabel('state of the pilot',fontsize = 12)

plt.ylabel('Count',fontsize = 12)

plt.title('Target repartition',fontsize = 15)

plt.show()
plt.figure(figsize = (15,10))

sns.countplot('experiment',hue = 'event',data = train_df)

plt.xlabel('experminet and state of the pilot',fontsize = 12)

plt.ylabel('Count (log)',fontsize = 12)

plt.yscale('log')

plt.title('Target reprtition for diffrent experminets',fontsize = 15)

plt.show()
plt.figure(figsize = (15,10))

sns.countplot('event',hue = 'seat',data = train_df)

plt.ylabel('seat and state of the pilot',fontsize = 12)

plt.ylabel('Count (log)',fontsize = 12)

plt.yscale('log')

plt.title('Left seat or right seat ?',fontsize = 15)

plt.show()
# Time expriment 



plt.figure(figsize = (15,10))

sns.violinplot(x = 'event',y = 'time',data = train_df.sample(50000))

plt.ylabel('Time (s)',fontsize = 12)

plt.xlabel('Event',fontsize = 12)

plt.title('Which time do event occur at',fontsize = 15)

plt.show()
plt.figure(figsize =(15,10))

sns.distplot(test_df['time'],label = 'Test set')

sns.distplot(train_df['time'],label = 'Train set')

plt.legend()

plt.xlabel('Time (s)',fontsize = 12)

plt.title('Reparition of the time feature',fontsize = 15)

plt.show()
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]
plt.figure(figsize =(20,25))

i = 0



for egg in eeg_features:

    i += 1

    plt.subplot(5,4,i)

    sns.boxplot(x = 'event', y = egg , data = train_df.sample(50000),showfliers = False)

    plt.show()
# Also check if features have the same distribution on the test and train set



plt.figure(figsize = (20,25))

plt.title('Eeg features distributions')

i = 0



for eeg in eeg_features:

    i += 1

    plt.subplot(5,4,i)

    sns.distplot(test_df.sample(10000)[eeg], label = 'Test set',hist = False)

    sns.distplot(train_df.sample(10000)[eeg],label = 'Train set',hist = False)

    plt.xlim((-500 , 500))

    plt.legend()

    plt.xlabel(egg , fontsize = 12)