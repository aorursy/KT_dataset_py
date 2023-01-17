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
dataset= "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"
dir_list = os.listdir(dataset)
dir_list[0:10]
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
    path.append(dataset + i)
emotion
class_col = pd.DataFrame(emotion, columns = ['class'])
class_col
class_col["class"].value_counts()
path
dataset_df=pd.DataFrame(path,columns=['path'])
dataset_df['source'] = 'SAVEE'
dataset_df
dataset_df['class']=class_col
dataset_df[0:10]
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd

#
fname = path[0]  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(5, 2.5))
librosa.display.waveplot(data, sr=sampling_rate)

# Lets play the audio 
ipd.Audio(fname)
fig, ax = plt.subplots(nrows=3,ncols=3, sharex=False, sharey=False,figsize=(25,15))

x=0
for i in range(3):
    for j in range(3):
        fname = path[x]  
        data, sampling_rate = librosa.load(fname)
        plt.figure(figsize=(25, 25))
        ax[i][j].set(title=dataset_df['class'][x])
        ax[i][j].set_ylabel('Amplitude Envelope')
        librosa.display.waveplot(data, sr=sampling_rate, ax=ax[i][j])
        x+=1
fig.suptitle('MONOPHONIC PLOT')
fig, ax = plt.subplots(nrows=3,ncols=3, sharex=False, sharey=False,figsize=(25,15))

x=0
for i in range(3):
    for j in range(3):
        fname = path[x]  
        data, sampling_rate = librosa.load((fname), mono=False, duration=10)
        plt.figure(figsize=(25, 25))
        ax[i][j].set(title=dataset_df['class'][x])
        ax[i][j].set_ylabel('Amplitude Envelope')
        librosa.display.waveplot(data, sr=sampling_rate, ax=ax[i][j])
        x+=1
fig.suptitle('STEREO PLOT')
fig, ax = plt.subplots(nrows=3,ncols=3, sharex=False, sharey=False,figsize=(25,15))

x=0
for i in range(3):
    for j in range(3):
        fname = path[x]  
        data, sampling_rate = librosa.load(fname)
        y_harm, y_perc = librosa.effects.hpss(data)
        
        plt.figure(figsize=(25, 25))
        ax[i][j].set(title=dataset_df['class'][x])
        ax[i][j].set_ylabel('Amplitude Envelope')
        librosa.display.waveplot(y_harm, sr=sampling_rate, alpha=0.25,ax=ax[i][j])
        librosa.display.waveplot(y_perc, sr=sampling_rate, ax=ax[i][j], alpha=0.5,color='r')
        
        x+=1
fig.suptitle('Harmonic + Percussive')
fig, ax = plt.subplots(nrows=3,ncols=3, sharex=False, sharey=False,figsize=(25,15))

x=0
for i in range(3):
    for j in range(3):
        fname = path[x]  
        data, sampling_rate = librosa.load((fname), mono=False, duration=10)
        plt.figure(figsize=(25, 25))
        ax[i][j].set(title=dataset_df['class'][x])
        ax[i][j].set_ylabel('Amplitude Envelope')
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sampling_rate, ax=ax[i][j])

        
        x+=1
fig.suptitle('Linear-frequency power spectrogram')
fig, ax = plt.subplots(nrows=3,ncols=3, sharex=False, sharey=False,figsize=(25,15))

x=0
hop_length = 1024
for i in range(3):
    for j in range(3):
        fname = path[x]  
        data, sampling_rate = librosa.load((fname), mono=False, duration=10)
        plt.figure(figsize=(25, 25))
        ax[i][j].set(title=dataset_df['class'][x])
        ax[i][j].set_ylabel('Amplitude Envelope')
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data, hop_length=hop_length)),
                            ref=np.max)
        img=librosa.display.specshow(D, y_axis='log', x_axis='time', hop_length=hop_length,
                               sr=sampling_rate, ax=ax[i][j])

        img
        x+=1
fig.suptitle('Log-frequency power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")
dataset_df
TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
CREMA = "/kaggle/input/cremad/AudioWAV/"
dir_list = os.listdir(RAV)

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

new_data1=pd.DataFrame(path,columns=['path'])
new_data1['source'] = 'RAVDESS'
RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
new_data1['class']=RAV_df.gender + '_' + RAV_df.emotion
new_data1
new_data1['class'].value_counts()
dir_list = os.listdir(TESS)
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

new_dataset2=pd.DataFrame(path,columns=['path'])
new_dataset2['source']='TESS'
new_dataset2['class']=pd.DataFrame(emotion)
new_dataset2
dir_list = os.listdir(CREMA)
dir_list
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
    
new_dataset3=pd.DataFrame(path,columns=['path'])
new_dataset3['source']='CREMA'
new_dataset3['class']=pd.DataFrame(emotion)
new_dataset3
dataset_df=pd.concat([dataset_df,new_data1,new_dataset2,new_dataset3],axis=0, ignore_index=True)
dataset_df
dataset_df_1=dataset_df
df = pd.DataFrame(columns=['feature'])
# loop feature extraction over the entire dataset
counter=0
for index,path in enumerate(dataset_df_1.path):
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  ,duration=2.5
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
dataset_df_1 = pd.concat([dataset_df_1,pd.DataFrame(df['feature'].values.tolist())],axis=1)
dataset_df_1[:5]
dataset_df_1.to_csv("Data_path_proc.csv",index=False)
dataset_df_1.fillna(0)
# Split between train and test 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(dataset_df_1.drop(['path','class','source'],axis=1)
                                                    , dataset_df_1['class']
                                                    , test_size=0.3
                                                    , shuffle=True
                                                    , random_state=42
                                                   )
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
#X_test = (X_test - mean)/std

X_train
y_train.shape
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
import pickle
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
filename = 'class'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_train.shape
