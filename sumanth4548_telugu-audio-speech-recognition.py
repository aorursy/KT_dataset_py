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
# Import libraries 

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys

import warnings

# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



#TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

#RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

SAVEE = "../input/speechdataset/my-Audio-Dataset/Emotions/Angry/"

#CREMA = "/kaggle/input/cremad/AudioWAV/"

# Run one example 

dir_list = os.listdir(SAVEE)

dir_list[0:16]
# Get the data location for SAVEE

dir_list = os.listdir(SAVEE)



# parse the filename to get the emotions

emotion=[]

path = []

for i in dir_list:

    if i[-11:-10]=='m':

        emotion.append('male_angry')

   

    elif i[-11:-10]=='f':

        emotion.append('Female_Angry')

   

   #elif i[-8:-6]=='sa':

   #     emotion.append('male_sad')

    else:

        emotion.append('error') 

    path.append(SAVEE + i)

# Now check out the label count distribution 

SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'

SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)

SAVEE_df.labels.value_counts()
# use the well known Librosa library for this task 

fname = SAVEE + 'Angry.01.male.01.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# Import our libraries

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import specgram

import pandas as pd

import os

import IPython.display as ipd  # To play sound in the notebook
# Source - RAVDESS; Gender - Female; Emotion - Angry 

path = "../input/speechdataset/my-Audio-Dataset/Emotions/Angry/Angry.02.male.01.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5) 

#path:string, int, pathlib.Path or file-like object

#res_type:str

#By default, this uses resampy’s high-quality mode (‘kaiser_best’).

#To use a faster method, set res_type=’kaiser_fast’.

#duration:float

#only load up to this much audio (in seconds)

#sr:number > 0 [scalar]

#target sampling rate

#Sampling rate or sampling frequency defines the number of samples per second (or per other unit) taken from a continuous signal to make a discrete or digital signal

#offset:float

#start reading after this time (in seconds)



mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

#y:np.ndarray [shape=(n,)] or None

#audio time series

#sr:number > 0 [scalar]

#sampling rate of y

#n_mfcc: int > 0 [scalar]

#number of MFCCs to return

#Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip





# audio wave

plt.figure(figsize=(20, 15))

#figsize(float, float), optional, default: None

#width, height in inches. If not provided, defaults to rcParams["figure.figsize"] (default: [6.4, 4.8]) = [6.4, 4.8]



plt.subplot(3,1,1)

#subplot(nrows, ncols, index, **kwargs)

#kwargs definition is available in https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html



librosa.display.waveplot(X, sr=sample_rate)

#X:np.ndarray [shape=(n,) or (2,n)]

#audio time series (mono or stereo)

#sr:number > 0 [scalar]

#sampling rate of X



plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Importing required libraries 

# Keras

import keras

from keras import regularizers

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.utils import np_utils, to_categorical

from keras.callbacks import ModelCheckpoint



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

import pickle

import IPython.display as ipd  # To play sound in the notebook
# lets pick up the meta-data that we got from our first part of the Kernel

ref = pd.read_csv("../input/pathsdatanew/Angry.csv")

ref.tail()
# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets 

df = pd.DataFrame(columns=['feature'])



# loop feature extraction over the entire dataset

counter=0

for index,path in enumerate(ref.path):

    X, sample_rate = librosa.load(path

                                  , res_type='kaiser_fast'

                                  ,duration=30

                                  ,sr=44100

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    # mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=sample_rate, 

                                        n_mfcc=13),

                    axis=0)

    df.loc[counter] = [mfccs]

    counter=counter+1   



# Check a few records to make sure its processed successfully

print(len(df))

df.head()