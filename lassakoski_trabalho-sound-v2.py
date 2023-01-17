# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

from memory_profiler import memory_usage

import os

from glob import glob



import IPython.display as ipd

import librosa

import librosa.display

import matplotlib.pyplot as plt

import os

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint 

from datetime import datetime 

from scipy.fftpack import fft,fftfreq



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Criar pastas para salvar as imagens de treinamento, validação e teste



!mkdir /kaggle/working/train

!mkdir /kaggle/working/valid

!mkdir /kaggle/working/test
dataset = pd.read_csv('../input/urbansound8k/UrbanSound8K.csv')



dataset_size = len(dataset)

train_size = int(dataset_size*0.7)

valid_size = int(dataset_size*0.15)

test_size = int(dataset_size*0.15)



print('train size: ' + str(train_size))

print('valid size: ' + str(valid_size))

print('test size: ' + str(test_size))



train_data = dataset[0:train_size].copy().reset_index(drop=True)

valid_data = dataset[train_size:(train_size+valid_size)].copy().reset_index(drop=True)

test_data = dataset[(train_size+valid_size):(train_size+valid_size+test_size)].copy().reset_index(drop=True)



print(train_data.shape)

print(valid_data.shape)

print(test_data.shape)
#pegar apenas nome do arquivo

train_data['slice_file_name'][0].split('.')[0]
#criar as imagens para treinamento

dir_img = '/kaggle/working/train/'

fulldatasetpath = '../input/urbansound8k/'

for index, row in tqdm(train_data.iterrows()):

    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))

    train_signals = librosa.load(filename)[0]  
#criar as imagens para validação

dir_img = '/kaggle/working/valid/'

fulldatasetpath = '../input/urbansound8k/'

for index, row in tqdm(valid_data.iterrows()):

    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))

    valid_signals = librosa.load(filename)[0]  
#criar as imagens para treinamento

dir_img = '/kaggle/working/test/'

fulldatasetpath = '../input/urbansound8k/'

for index, row in tqdm(test_data.iterrows()):

    filename = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))

    test_signals = librosa.load(filename)[0]  
def extract_features(signal):

    import numpy as np

    return [

        librosa.feature.zero_crossing_rate(signal)[0, 0],

        librosa.feature.spectral_centroid(signal)[0, 0],

    ]
train_features = np.array([extract_features(x) for x in train_signals]) 

valid_features = np.array([extract_features(x) for x in valid_signals]) 

test_features = np.array([extract_features(x) for x in test_signals]) 