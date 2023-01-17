# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

import librosa as lb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
diagnosis=pd.read_csv('/kaggle/input/respiratory-sound-database/respiratory_sound_database/Respiratory_Sound_Database/patient_diagnosis.csv',names=['pid','disease'])

diagnosis.head()
sns.countplot(diagnosis.disease)

plt.xticks(rotation=90)
import os

def extractId(filename):

    return filename.split('_')[0]
path='/kaggle/input/preprocessing-part-1/processed_audio_files/'

length=len(os.listdir(path))

index=range(length)

i=0

files_df=pd.DataFrame(index=index,columns=['pid','filename'])

for f in os.listdir(path):

    files_df.iloc[i]['pid']=extractId(f)

    files_df.iloc[i]['filename']=f

    i+=1

files_df.head()
files_df.pid=files_df.pid.astype('int64') # both pid's must be of same dtype for them to merge
data=pd.merge(files_df,diagnosis,on='pid')

data.head()
sns.countplot(data.disease)

plt.xticks(rotation=90)
from sklearn.model_selection import train_test_split

Xtrain,Xval,ytrain,yval=train_test_split(data,data.disease,stratify=data.disease,random_state=42,test_size=0.25)
Xtrain.disease.value_counts()/Xtrain.shape[0]
Xval.disease.value_counts()/Xval.shape[0]
path='../input/preprocessing-part-1/processed_audio_files/'



import librosa.display

file=path + Xtrain.iloc[193].filename 

sound,sample_rate=lb.load(file)

mfccs = lb.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40)

fig, ax = plt.subplots()

img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)

fig.colorbar(img, ax=ax)

ax.set(title='MFCC')
Xtrain.to_csv('train.csv')

Xval.to_csv('val.csv')