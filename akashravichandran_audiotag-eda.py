import numpy as np 

import pandas as pd 

import os

import shutil

import wave

import IPython

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook

import sklearn

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

SAMPLE_RATE = 44100

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

matplotlib.style.use('ggplot')

print(os.listdir("../input"))
train = pd.read_csv("../input/train_curated.csv")

train_noisy = pd.read_csv("../input/train_noisy.csv")

test = pd.read_csv("../input/sample_submission.csv")



def explode_str(df, col, sep):

    s = df[col]

    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)

    return df.iloc[i].assign(**{col: sep.join(s).split(sep)})



# def explode_list(df, col):

#     s = df[col]

#     i = np.arange(len(s)).repeat(s.str.len())

#     return df.iloc[i].assign(**{col: np.concatenate(s)})



def load_wav_file(name, path):

    _, b = wavfile.read(path + name)

    assert _ == SAMPLE_RATE

    return b
print('Train Curated :')

train.head(10)
# %%time

# pd.concat([pd.Series(row['fname'], row['labels'].split(','))              

#                     for _, row in train.iterrows()]).reset_index()

# CPU times: user 2.27 s, sys: 76 ms, total: 2.35 s

# Wall time: 2.33 s

ntrain = explode_str(train, 'labels', ',')

print('Train Curated after exploding :')

ntrain.head(10)
print("Number of curated training examples=", train.shape[0], "  Number of curated training classes=", len(train.labels.unique()))

print("Number of curated training examples after exploding=", ntrain.shape[0], "  Number of curated training classes after exploding=", len(ntrain.labels.unique()))
# pd.DataFrame({'unique_train_labels':ntrain.labels.unique()})

# ntrain.labels.unique()

print("Total number of labels in curated training data : ",len(ntrain['labels'].value_counts()))

print("Labels are : ", ntrain['labels'].unique())

plt.figure(figsize=(10,6))

audio_type = ntrain['labels'].value_counts().head(10)

sns.barplot(audio_type.values, audio_type.index)

for i, v in enumerate(audio_type.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xticks(rotation='vertical')

plt.xlabel('Frequency')

plt.ylabel('Label Name')

plt.title("First few labels based on their frequencies in curated training data")

plt.show()
plt.figure(figsize=(10,6))

naudio_type = ntrain['labels'].value_counts().tail(10)

sns.barplot(naudio_type.values, naudio_type.index)

for i, v in enumerate(naudio_type.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xticks(rotation='vertical')

plt.xlabel('Frequency')

plt.ylabel('Label Name')

plt.title("Last few labels based on their frequencies in curated training data")

plt.show()
INPUT_LIB = '../input/'

new_train = ntrain.sort_values('labels').reset_index()

new_train['nframes'] = new_train['fname'].apply(lambda f: wave.open('../input/train_curated/' + f).getnframes())



new_train['series'] = new_train['fname'].apply(load_wav_file, 

                                                      path=INPUT_LIB + 'train_curated/')



_, ax = plt.subplots(figsize=(18, 5))

sns.violinplot(ax=ax, x="labels", y="nframes", data=new_train)

plt.xticks(rotation=90)

plt.title('Distribution of audio frames, per label in train curated', fontsize=16)

plt.show()
print('Histogram of nframes with respect to Train Curated :')

plt.figure(figsize=(12,8))

sns.distplot(new_train.nframes.values, bins=50, kde=False)

plt.xlabel('nframes', fontsize=12)

plt.title("Histogram of #frames")

plt.show()
print('We can see an outlier in the above plot which belongs to the label - Stream')

new_train.loc[new_train['nframes'] > 2000000]
print('Temporary data for series plotting :')

temp = new_train.sort_values(by='labels')

temp.head()
print("Accelerating_and_revving_and_vroom : ")

fig, ax = plt.subplots(10, 4, figsize = (12, 16))

for i in range(40):

    ax[i//4, i%4].plot(temp['series'][i])

    ax[i//4, i%4].set_title(temp['fname'][i][:-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])

fig.savefig("Accelerating_and_revving_and_vroom", dpi=900) 
from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=800, height=500).generate(' '.join(new_train.labels))

plt.figure(figsize=(18,10))

plt.imshow(wordcloud)

plt.title("Wordcloud for Labels in Train Curated", fontsize=25)

plt.axis("off")

plt.show()
print('Train Noisy :')

train_noisy.head(10)
print('Train Noisy after exploding :')

ntrain_noisy = explode_str(train_noisy, 'labels', ',')

ntrain_noisy.head(10)
print("Number of noisy training examples=", train_noisy.shape[0], "  Number of noisy training classes=", len(train_noisy.labels.unique()))

print("Number of noisy training examples after exploding=", ntrain_noisy.shape[0], "  Number of noisy training classes after exploding=", len(ntrain_noisy.labels.unique()))
# pd.DataFrame({'unique_noisytrain_labels':ntrain_noisy.labels.unique()})

# ntrain_noisy.labels.unique()

print("Total number of labels in curated training data : ",len(ntrain_noisy['labels'].value_counts()))

print("Labels are : ", ntrain_noisy['labels'].unique())

plt.figure(figsize=(10,6))

audio_type = ntrain_noisy['labels'].value_counts().head(10)

sns.barplot(audio_type.values, audio_type.index)

for i, v in enumerate(audio_type.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xticks(rotation='vertical')

plt.xlabel('Frequency')

plt.ylabel('Label Name')

plt.title("First few labels based on their frequencies in noisy training data")

plt.show()
plt.figure(figsize=(10,6))

naudio_type = ntrain_noisy['labels'].value_counts().tail(10)

sns.barplot(naudio_type.values, naudio_type.index)

for i, v in enumerate(naudio_type.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xticks(rotation='vertical')

plt.xlabel('Frequency')

plt.ylabel('Label Name')

plt.title("Last few labels based on their frequencies in noisy training data")

plt.show()
new_noisytrain = ntrain_noisy.sort_values('labels').reset_index()

new_noisytrain['nframes'] = new_noisytrain['fname'].apply(lambda f: wave.open('../input/train_noisy/' + f).getnframes())

# new_noisytrain['series'] = new_noisytrain['fname'].apply(load_wav_file, 

#                                                       path=INPUT_LIB + 'train_noisy/')

# new_noisytrain['nframes'] = new_noisytrain['series'].apply(len)

_, ax = plt.subplots(figsize=(18, 5))

sns.violinplot(ax=ax, x="labels", y="nframes", data=new_noisytrain)

plt.xticks(rotation=90)

plt.title('Distribution of audio frames, per label in train noisy', fontsize=16)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(new_noisytrain.nframes.values, bins=50, kde=False)

plt.xlabel('nframes', fontsize=12)

plt.title("Histogram of #frames")

plt.show()
from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=800, height=500).generate(' '.join(new_noisytrain.labels))

plt.figure(figsize=(18,10))

plt.imshow(wordcloud)

plt.title("Wordcloud for Labels in Train Curated", fontsize=30)

plt.axis("off")

plt.show()