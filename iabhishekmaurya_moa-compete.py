import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_tragets_sccored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_tragets_sccored.head()
train_tragets_sccored.sum()[1:].sort_values()
train_features.shape
train_features.head()
train_features['cp_type'].value_counts()
train_features['cp_time'].value_counts()
train_features['cp_dose'].value_counts()