import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import glob

import librosa

import librosa.display

from tqdm import tqdm_notebook as tqdm

from keras.models import Model

from keras.utils import np_utils



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline 

import matplotlib.pyplot as plt
LIMIT = 3
!ls ../input/birdsong-recognition
df_train = pd.read_csv('../input/birdsong-recognition/train.csv')

df_train
!ls ../input/birdsong-recognition/train_audio



train_dir = '../input/birdsong-recognition/train_audio'

test_idr = '../input/birdsong-recognition/test_audio'
def mfcc_extract(filename):

    try:

        y, sr  = librosa.load(filename, sr = 44100)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(0.02*sr),hop_length=int(0.01*sr))

        return mfcc

    except:

        return
def parse_audio_files(parent_dir, sub_dirs, limit):

    labels = []

    features = []

    for label, sub_dir in enumerate(tqdm(sub_dirs)):

        i = 0

        for fn in glob.glob(os.path.join(parent_dir,sub_dir,"*.mp3")):

            if i >= limit:

                break

            features.append(mfcc_extract(fn))

            labels.append(label)

            i+=1

    return features, labels
%%time



train_cat_dirs = glob.glob(train_dir+'/*')

train_cat = []

for cat_dir in train_cat_dirs:

    tmp = cat_dir.split('/')[-1]

    train_cat.append(tmp)

print('the number of kinds:', len(train_cat))



class_num = len(train_cat)

features, labels = parse_audio_files(train_dir, train_cat, LIMIT)
print(len(features))

print(features[0].shape)
# plot few features



fig = plt.figure(figsize=(28,24))

for i,mfcc in enumerate(tqdm(features[:100])):

    if i%40 < 3 : 

        sub = plt.subplot(10,3,i%40+3*(i/40)+1)

        librosa.display.specshow(mfcc,vmin=-700,vmax=300)

        if ((i%40+3*(i/40)+1)%3==0) : 

            plt.colorbar()

        sub.set_title(train_cat[labels[i]])

plt.show()  
df_submission = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

df_submission.to_csv('submission.csv', index = None)