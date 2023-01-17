# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_features.head(5)
train_features.values.shape
train_features.columns
# Key features

print("C-features (Cell viability features): ",len([i for i in train_features.columns if 'c-' in i]))

print("G-features (Gene expression features): ",len([i for i in train_features.columns if 'g-' in i]))

print("Key features:")

train_features[['cp_type','cp_time','cp_dose']]
# N/a values?

for column in train_features.columns:

    if train_features[column].isna().sum() > 0: print(column)
# Doesn't seem like there are any.

# Feature visualization

from matplotlib import pyplot as plt

%matplotlib inline

import random



for index, feature in enumerate(['cp_time','cp_type','cp_dose']):

    plt.hist(train_features[feature], color = 'bgrcmykw'[index])

    plt.title(feature)

    plt.show()
# What do our targets look like?

targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

targets_scored.drop('sig_id',axis=1, inplace=True)

targets_scored.head(5)
d = {}

for moa in targets_scored.columns:

    d[moa] = sum(targets_scored[moa])/len(targets_scored[moa])

sorted_d = list(reversed(sorted(d.items(), key=lambda kv: kv[1])))

sorted_d[:20]
fig = plt.figure(figsize=(30, 10))

plt.bar(x=[i[0] for i in sorted_d[:10]], height = [i[1] for i in sorted_d[:10]])

plt.show()
# Submit submission
# Benchmark

import copy

pred_input = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

submission_data = np.zeros((pred_input.values.shape[0],targets_scored.values.shape[1])).astype(str)

for index, feature in enumerate(targets_scored.columns[1:]):

    submission_data[:,index+1] = str(d[feature])

submission_data[:,0] = pred_input['sig_id'].values
submission_df = pd.DataFrame(data=submission_data, columns=targets_scored.columns)

submission_df.head()
submission_df.to_csv('submission.csv',index=False)
with open('submission.csv', 'r') as reader:

    p = reader.readlines()[:2]

p