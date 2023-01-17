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
TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

CREMA = "/kaggle/input/cremad/AudioWAV/"



dir_list = os.listdir(SAVEE)

dir_list[0:5]
# Get the data location for SAVEE

dir_list = os.listdir(SAVEE)



# parse the filename to get the emotions

emotion=[]

path = []

for i in dir_list:

    if i[-8:-6]=='_a':

        emotion.append('male_angry')

    elif i[-8:-6]=='_d':

        emotion.append('male_disgust')

    elif i[-8:-6]=='_f':

        emotion.append('male_fear')

    elif i[-8:-6]=='_h':

        emotion.append('male_happy')

    elif i[-8:-6]=='_n':

        emotion.append('male_neutral')

    elif i[-8:-6]=='sa':

        emotion.append('male_sad')

    elif i[-8:-6]=='su':

        emotion.append('male_surprise')

    else:

        emotion.append('male_error') 

    path.append(SAVEE + i)

    

# Now check out the label count distribution 

SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'

SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)

SAVEE_df.labels.value_counts()
# use the well known Librosa library for this task 

fname = SAVEE + 'DC_f11.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# Lets play a happy track

fname = SAVEE + 'DC_h11.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
dir_list = os.listdir(RAV)

dir_list.sort()



emotion = []

gender = []

path = []

for i in dir_list:

    fname = os.listdir(RAV + i)

    for f in fname:

        part = f.split('.')[0].split('-')

        emotion.append(int(part[2]))

        temp = int(part[6])

        if temp%2 == 0:

            temp = "female"

        else:

            temp = "male"

        gender.append(temp)

        path.append(RAV + i + '/' + f)



        

RAV_df = pd.DataFrame(emotion)

RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})

RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)

RAV_df.columns = ['gender','emotion']

RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion

RAV_df['source'] = 'RAVDESS'  

RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)

RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)

RAV_df.labels.value_counts()
# Pick a fearful track

fname = RAV + 'Actor_14/03-01-06-02-02-02-14.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# Pick a happy track

fname = RAV + 'Actor_14/03-01-03-02-02-02-14.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
dir_list = os.listdir(TESS)

dir_list.sort()

dir_list
path = []

emotion = []



for i in dir_list:

    fname = os.listdir(TESS + i)

    for f in fname:

        if i == 'OAF_angry' or i == 'YAF_angry':

            emotion.append('female_angry')

        elif i == 'OAF_disgust' or i == 'YAF_disgust':

            emotion.append('female_disgust')

        elif i == 'OAF_Fear' or i == 'YAF_fear':

            emotion.append('female_fear')

        elif i == 'OAF_happy' or i == 'YAF_happy':

            emotion.append('female_happy')

        elif i == 'OAF_neutral' or i == 'YAF_neutral':

            emotion.append('female_neutral')                                

        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':

            emotion.append('female_surprise')               

        elif i == 'OAF_Sad' or i == 'YAF_sad':

            emotion.append('female_sad')

        else:

            emotion.append('Unknown')

        path.append(TESS + i + "/" + f)



TESS_df = pd.DataFrame(emotion, columns = ['labels'])

TESS_df['source'] = 'TESS'

TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)

TESS_df.labels.value_counts()
# lets play a fearful track 

fname = TESS + 'YAF_fear/YAF_dog_fear.wav' 



data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# lets play a happy track 

fname =  TESS + 'YAF_happy/YAF_dog_happy.wav' 



data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
dir_list = os.listdir(CREMA)

dir_list.sort()

print(dir_list[0:10])
gender = []

emotion = []

path = []

female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,

          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]



for i in dir_list: 

    part = i.split('_')

    if int(part[0]) in female:

        temp = 'female'

    else:

        temp = 'male'

    gender.append(temp)

    if part[2] == 'SAD' and temp == 'male':

        emotion.append('male_sad')

    elif part[2] == 'ANG' and temp == 'male':

        emotion.append('male_angry')

    elif part[2] == 'DIS' and temp == 'male':

        emotion.append('male_disgust')

    elif part[2] == 'FEA' and temp == 'male':

        emotion.append('male_fear')

    elif part[2] == 'HAP' and temp == 'male':

        emotion.append('male_happy')

    elif part[2] == 'NEU' and temp == 'male':

        emotion.append('male_neutral')

    elif part[2] == 'SAD' and temp == 'female':

        emotion.append('female_sad')

    elif part[2] == 'ANG' and temp == 'female':

        emotion.append('female_angry')

    elif part[2] == 'DIS' and temp == 'female':

        emotion.append('female_disgust')

    elif part[2] == 'FEA' and temp == 'female':

        emotion.append('female_fear')

    elif part[2] == 'HAP' and temp == 'female':

        emotion.append('female_happy')

    elif part[2] == 'NEU' and temp == 'female':

        emotion.append('female_neutral')

    else:

        emotion.append('Unknown')

    path.append(CREMA + i)

    

CREMA_df = pd.DataFrame(emotion, columns = ['labels'])

CREMA_df['source'] = 'CREMA'

CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)

CREMA_df.labels.value_counts()
# use the well known Librosa library for this task 

fname = CREMA + '1012_IEO_HAP_HI.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
# A fearful track

fname = CREMA + '1012_IEO_FEA_HI.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Lets play the audio 

ipd.Audio(fname)
df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)

print(df.labels.value_counts())

df.head()

df.to_csv("Data_path.csv",index=False)