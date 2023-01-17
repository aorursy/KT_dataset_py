import numpy as np

import pandas as pd



import os

import IPython.display as ipd

pd.set_option('max_columns', 50)

pd.set_option('max_rows', 150)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
os.listdir('/kaggle/input/birdsong-recognition')
len(os.listdir('/kaggle/input/birdsong-recognition/train_audio'))
ipd.Audio('/kaggle/input/birdsong-recognition/train_audio/nutwoo/XC462016.mp3')
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

train.shape
train.head()
train['ebird_code'].nunique()
plt.figure(figsize=(12, 8))

train['ebird_code'].value_counts().plot(kind='hist')
train['location'].value_counts()
train['location'].apply(lambda x: x.split(',')[-1]).value_counts().head(10)
train['location'].value_counts().plot(kind='hist')
train['location'].nunique()
plt.figure(figsize=(12, 8))

train['country'].value_counts().head(20).plot(kind='barh');
plt.figure(figsize=(20, 8))

train['date'].value_counts().sort_index().plot();
train['date'].sort_values()[15:30].values
train['rating'].value_counts().plot(kind='barh')

plt.title('Counts of different ratings');
fig, ax = plt.subplots(figsize=(24, 6))

plt.subplot(1, 2, 1)

train.groupby(['ebird_code']).agg({'rating': ['mean', 'std']}).reset_index().sort_values(('rating', 'mean'), ascending=False).set_index('ebird_code')['rating']['mean'].plot(kind='bar')

plt.subplot(1, 2, 2)

train.groupby(['ebird_code']).agg({'rating': ['mean', 'std']}).reset_index().sort_values(('rating', 'mean'), ascending=False).set_index('ebird_code')['rating']['mean'][:20].plot(kind='barh')
train['duration'].plot(kind='hist')

plt.title('Distribution of durations');
for i in range(50, 100, 5):

    perc = np.percentile(train['duration'], i)

    print(f"{i} percentile of duration is {perc}")
test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')

test
test_metadata = pd.read_csv('/kaggle/input/birdsong-recognition/example_test_audio_metadata.csv')

test_metadata.shape
test_metadata.head()
test_summary = pd.read_csv('/kaggle/input/birdsong-recognition/example_test_audio_summary.csv')

test_summary.head()
test_summary.shape
sub = pd.read_csv('/kaggle/input/birdsong-recognition/sample_submission.csv')

sub
sub.to_csv('submission.csv', index=False)