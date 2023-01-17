%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import gc

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import seaborn as sns

sns.set(font_scale=1.2)



import warnings



pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 100)

# dir(pd.options.display)

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=UserWarning)



plt.style.use('ggplot')



import missingno as msno

import datetime as dt

import IPython.display as ipd

import librosa



from random import sample

from collections import OrderedDict

from datetime import timedelta

from mpl_toolkits.axes_grid1 import make_axes_locatable



np.random.seed(2019)
SAMPLE_RATE = 44100
train = pd.read_csv("../input/train_curated.csv")

train['is_curated'] = True

train_noisy = pd.read_csv('../input/train_noisy.csv')

train_noisy['is_curated'] = False

train = pd.concat([train, train_noisy], axis=0)

del train_noisy
train.sample(5)
train['n_label'] = train.labels.str.split(',').apply(lambda x: len(x))
train.query('is_curated == True').n_label.value_counts()
train.query('is_curated == False').n_label.value_counts()
cat_gp = train[train.n_label == 1].groupby(

    ['labels', 'is_curated']).agg({

    'fname':'count'

}).reset_index()

cat_gpp = cat_gp.pivot(index='labels', columns='is_curated', values='fname').reset_index().set_index('labels')



plot = cat_gpp.plot(

    kind='barh',

    title="Number of samples per category",

    stacked=True,

    color=['deeppink', 'darkslateblue'],

    figsize=(15,20))

plot.set_xlabel("Number of Samples", fontsize=20)

plot.set_ylabel("Label", fontsize=20);
# sampling an audio in train_curated

samp = train[(train.n_label == 1) & (train.is_curated == True)].sample(1)

print(samp.labels.values[0])

ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))
# sampling an audio in train_noisy

samp_n = train[(train.n_label == 1) & 

               (train.is_curated == False) & 

               (train.labels == samp.labels.values[0])].sample(1)

print(samp_n.labels.values[0])

ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))
# trim silent part

wav, _ = librosa.core.load(

    '../input/train_curated/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

wav_n, _ = librosa.core.load(

    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr_n = librosa.effects.trim(wav_n)[0]

print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))

print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))
melspec = librosa.feature.melspectrogram(

    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel = librosa.core.power_to_db(melspec)



melspec_n = librosa.feature.melspectrogram(

    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel_n = librosa.core.power_to_db(melspec_n)
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i, l in enumerate([logmel, logmel_n]):

    if i==0: 

        ax[i].set_title('curated {}'.format(samp.labels.values[0]))

    else:

        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

    im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',

                      aspect=l.shape[1]/l.shape[0]/5)
mfcc = librosa.feature.mfcc(wav_tr, 

                            sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

mfcc_n = librosa.feature.mfcc(wav_tr_n, 

                              sr=SAMPLE_RATE, 

                              n_fft=1764,

                              hop_length=220,

                              n_mfcc=64)



fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i, m in enumerate([mfcc, mfcc_n]):

    if i==0: 

        ax[i].set_title('curated {}'.format(samp.labels.values[0]))

    else:

        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',

                      aspect=m.shape[1]/m.shape[0]/5)
print('Unique number of multi label : {}'.format(train.loc[train.n_label > 1, 'labels'].nunique()))

print('Unique number of multi label in curated data : {}'.format(

    train.loc[(train.n_label > 1) & (train.is_curated == True), 'labels'].nunique()))

print('Unique number of multi label in noisy data : {}'.format(

    train.loc[(train.n_label > 1) & (train.is_curated == False), 'labels'].nunique()))    
cat_gp = train[(train.n_label > 1) & (train.is_curated == True)].groupby(

    'labels').agg({'fname':'count'})

cat_gp.columns = ['counts']



plot = cat_gp.sort_values(ascending=True, by='counts').plot(

    kind='barh',

    title="Number of Audio Samples per Category",

    color='deeppink',

    figsize=(15,30))

plot.set_xlabel("Number of Samples", fontsize=20)

plot.set_ylabel("Label", fontsize=20);
label_set = set(train.loc[(train.n_label == 2) & (train.is_curated == True), 'labels']) & set(

    train.loc[(train.n_label == 2) & (train.is_curated == False), 'labels'])



label_samp = np.random.choice(list(label_set), 1)[0]

samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)

print(label_samp)

ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))
# sampling an audio in train_noisy

samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)

print(samp_n.labels.values[0])

ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))
# trim silent part

wav, _ = librosa.core.load(

    '../input/train_curated/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))



wav_n, _ = librosa.core.load(

    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr_n = librosa.effects.trim(wav_n)[0]

print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))
melspec = librosa.feature.melspectrogram(

    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel = librosa.core.power_to_db(melspec)



melspec_n = librosa.feature.melspectrogram(

    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel_n = librosa.core.power_to_db(melspec_n)
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

if samp_n.labels.values[0] == samp.labels.values[0]:

    for i, l in enumerate([logmel, logmel_n]):

        if i==0: 

            ax[i].set_title('curated {}'.format(samp.labels.values[0]))

        else:

            ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

        im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',

                          aspect=l.shape[1]/l.shape[0]/5)

mfcc = librosa.feature.mfcc(wav_tr, 

                            sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

mfcc_n = librosa.feature.mfcc(wav_tr_n, 

                              sr=SAMPLE_RATE, 

                              n_fft=1764,

                              hop_length=220,

                              n_mfcc=64)



fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i, m in enumerate([mfcc, mfcc_n]):

    if i==0: 

        ax[i].set_title('curated {}'.format(samp.labels.values[0]))

    else:

        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',

                      aspect=m.shape[1]/m.shape[0]/5)
label_set = set(train.loc[(train.n_label == 3) & (train.is_curated == True), 'labels']) & set(

    train.loc[(train.n_label == 3) & (train.is_curated == False), 'labels'])



label_samp = np.random.choice(list(label_set), 1)[0]

samp = train[(train.labels == label_samp) & (train.is_curated == True)].sample(1)

print(label_samp)

ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))
# sampling an audio in train_noisy

samp_n = train[(train.labels == label_samp) & (train.is_curated == False)].sample(1)

print(samp_n.labels.values[0])

ipd.Audio('../input/train_noisy/{}'.format(samp_n.fname.values[0]))
# trim silent part

wav, _ = librosa.core.load(

    '../input/train_curated/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed curated wav: {:,}/{:,}'.format(len(wav_tr), len(wav)))



wav_n, _ = librosa.core.load(

    '../input/train_noisy/{}'.format(samp_n.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr_n = librosa.effects.trim(wav_n)[0]

print('After trimmed noisy wav: {:,}/{:,}'.format(len(wav_tr_n), len(wav_n)))
melspec = librosa.feature.melspectrogram(

    librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel = librosa.core.power_to_db(melspec)



melspec_n = librosa.feature.melspectrogram(

    librosa.resample(wav_tr_n, SAMPLE_RATE, SAMPLE_RATE/2),

    sr=SAMPLE_RATE/2,

    n_fft=1764,

    hop_length=220,

    n_mels=64

)

logmel_n = librosa.core.power_to_db(melspec_n)
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

if samp_n.labels.values[0] == samp.labels.values[0]:

    for i, l in enumerate([logmel, logmel_n]):

        if i==0: 

            ax[i].set_title('curated {}'.format(samp.labels.values[0]))

        else:

            ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

        im = ax[i].imshow(l, cmap='Spectral', interpolation='nearest',

                          aspect=l.shape[1]/l.shape[0]/5)

mfcc = librosa.feature.mfcc(wav_tr, 

                            sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

mfcc_n = librosa.feature.mfcc(wav_tr_n, 

                              sr=SAMPLE_RATE, 

                              n_fft=1764,

                              hop_length=220,

                              n_mfcc=64)



fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i, m in enumerate([mfcc, mfcc_n]):

    if i==0: 

        ax[i].set_title('curated {}'.format(samp.labels.values[0]))

    else:

        ax[i].set_title('noisy {}'.format(samp_n.labels.values[0]))

    im = ax[i].imshow(m, cmap='Spectral', interpolation='nearest',

                      aspect=m.shape[1]/m.shape[0]/5)
samp = train[(train.n_label == 4) & (train.is_curated == True)].sample(1)

print(samp.labels.values[0])

ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))
wav, _ = librosa.core.load(

    '../input/train_curated/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))
wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)

melspec = librosa.feature.melspectrogram(wav_tr,

                                         sr=SAMPLE_RATE/2,

                                         n_fft=1764,

                                         hop_length=220,

                                         n_mels=64)

logmel = librosa.core.power_to_db(melspec)
fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',

          aspect=logmel.shape[1]/logmel.shape[0]/5)

ax.set_title('log-mel');
mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',

          aspect=mfcc.shape[1]/mfcc.shape[0]/5)

ax.set_title('MFCC');
samp = train[train.n_label == 5].sample(1)

print(samp.labels.values[0])

ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0]))
wav, _ = librosa.core.load(

    '../input/train_noisy/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))
wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)

melspec = librosa.feature.melspectrogram(wav_tr,

                                         sr=SAMPLE_RATE/2,

                                         n_fft=1764,

                                         hop_length=220,

                                         n_mels=64)

logmel = librosa.core.power_to_db(melspec)
fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',

          aspect=logmel.shape[1]/logmel.shape[0]/5)

ax.set_title('log-mel');
mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',

          aspect=mfcc.shape[1]/mfcc.shape[0]/5)

ax.set_title('MFCC');
samp = train[(train.n_label == 6) & (train.is_curated == True)].sample(1)

print(samp.labels.values[0])

ipd.Audio('../input/train_curated/{}'.format(samp.fname.values[0]))
wav, _ = librosa.core.load(

    '../input/train_curated/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))
wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)

melspec = librosa.feature.melspectrogram(wav_tr,

                                         sr=SAMPLE_RATE/2,

                                         n_fft=1764,

                                         hop_length=220,

                                         n_mels=64)

logmel = librosa.core.power_to_db(melspec)
fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',

          aspect=logmel.shape[1]/logmel.shape[0]/5)

ax.set_title('log-mel');
mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',

          aspect=mfcc.shape[1]/mfcc.shape[0]/5)

ax.set_title('MFCC');
samp = train[train.n_label == 7].sample(1)

print(samp.labels.values[0])

ipd.Audio('../input/train_noisy/{}'.format(samp.fname.values[0]))
wav, _ = librosa.core.load(

    '../input/train_noisy/{}'.format(samp.fname.values[0]),

    sr=SAMPLE_RATE)

wav_tr = librosa.effects.trim(wav)[0]

print('After trimmed: {:,}/{:,}'.format(len(wav_tr), len(wav)))
wav_tr = librosa.resample(wav_tr, SAMPLE_RATE, SAMPLE_RATE/2)

melspec = librosa.feature.melspectrogram(wav_tr,

                                         sr=SAMPLE_RATE/2,

                                         n_fft=1764,

                                         hop_length=220,

                                         n_mels=64)

logmel = librosa.core.power_to_db(melspec)
fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(logmel, cmap='Spectral', interpolation='nearest',

          aspect=logmel.shape[1]/logmel.shape[0]/5)

ax.set_title('log-mel');
mfcc = librosa.feature.mfcc(wav_tr, sr=SAMPLE_RATE, 

                            n_fft=1764,

                            hop_length=220,

                            n_mfcc=64)

fig, ax = plt.subplots(figsize=(15, 5))

im = ax.imshow(mfcc, cmap='Spectral', interpolation='nearest',

          aspect=mfcc.shape[1]/mfcc.shape[0]/5)

ax.set_title('MFCC');
test = pd.read_csv('../input/sample_submission.csv')

for c in test.columns[1:]:

    cc = c.replace('(', '\(').replace(')', '\)')

    train.loc[:, c] = train['labels'].str.contains(cc)

    if (train.loc[:, c] > 1).sum():

        raise Exception('label key "{}" are duplicated !'.format(c))
train.head()
tmp = train.loc[train.is_curated == True, 

                'Accelerating_and_revving_and_vroom':].sum(axis=0).to_frame().T

tmp = pd.concat(

    [tmp, train.loc[train.is_curated == False, 

                    'Accelerating_and_revving_and_vroom':].sum(axis=0).to_frame().T]

)

tmp['total_label'] = tmp.loc[:, 'Accelerating_and_revving_and_vroom':].sum(axis=1)

tmp.index = ['curated', 'noisy']

tmp
fig, ax = plt.subplots(figsize=(10, 20))

tmp.iloc[:, :-1].T.plot.barh(color=['deeppink', 'darkslateblue'], ax=ax);
sns.set(style="white")

sns.set(font_scale=1)

train_cor = train.loc[train.is_curated == True, test.columns[1:]].corr()

mask = np.zeros_like(train_cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



fig, ax = plt.subplots(figsize=(25, 25))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(train_cor, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .1});
multi_label = test.columns[1:][train.loc[

    (train.n_label > 1) & (train.is_curated == True), test.columns[1:]].sum() > 0]
sns.set(font_scale=1.7)

train_cor = train.loc[

    (train.n_label > 1) & (train.is_curated == True), multi_label].corr()

mask = np.zeros_like(train_cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.clustermap(train_cor, metric='correlation',

               cmap=cmap, center=0, linewidths=.5, figsize=(30, 30));