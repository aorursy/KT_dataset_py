# Basic Libraries

import pandas as pd

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



from sklearn.preprocessing import MinMaxScaler
#Libraries for classfication and building model

from tensorflow.keras.models import Sequential



from tensorflow.keras.layers import Conv2D ,Flatten,Dense,MaxPool2D,Dropout

from tensorflow.keras.utils import to_categorical



from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
#Project specific libraries

import os

import librosa

import librosa.display

import glob

import skimage
df = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")

'''We will extract classes from this metadata'''

df.head()
df.info()
df.describe()
df['fsID'].value_counts()
df['slice_file_name'].value_counts()
dat1,sampling_rate1=librosa.load('../input/urbansound8k/fold5/100032-3-0-0.wav')

dat2,sampling_rate2=librosa.load('../input/urbansound8k/fold5/100263-2-0-121.wav')



plt.figure(figsize=(20,10))

D=librosa.amplitude_to_db(np.abs(librosa.stft(dat1)),ref=np.max)

plt.subplot(4,2,1)

librosa.display.specshow(D,y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title('Linear-frequency power spectrogram')
plt.figure(figsize=(20,10))

D=librosa.amplitude_to_db(np.abs(librosa.stft(dat2)),ref=np.max)

plt.subplot(4,2,1)

librosa.display.specshow(D,y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title('Linear-frequency power spectrogram')