# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import seaborn as sns

import librosa.display

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
patient_data=pd.read_csv('/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv',names=['pid','disease'])
patient_data.head()
df=pd.read_csv('/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/160_1b3_Al_mc_AKGC417L.txt',sep='\t')

df.head()
import os

path='/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'

files=[s.split('.')[0] for s in os.listdir(path) if '.txt' in s]

files[:5]
def getFilenameInfo(file):

    return file.split('_')
getFilenameInfo('160_1b3_Al_mc_AKGC417L')
files_data=[]

for file in files:

    data=pd.read_csv(path + file + '.txt',sep='\t',names=['start','end','crackles','weezels'])

    name_data=getFilenameInfo(file)

    data['pid']=name_data[0]

    data['mode']=name_data[-2]

    data['filename']=file

    files_data.append(data)

files_df=pd.concat(files_data)

files_df.reset_index()

files_df.head()
patient_data.info()
files_df.info()
patient_data.pid=patient_data.pid.astype('int32')

files_df.pid=files_df.pid.astype('int32')
data=pd.merge(files_df,patient_data,on='pid')

data.head()
os.makedirs('csv_data')

data.to_csv('csv_data/data.csv',index=False)
def getPureSample(raw_data,start,end,sr=22050):

    '''

    Takes a numpy array and spilts its using start and end args

    

    raw_data=numpy array of audio sample

    start=time

    end=time

    sr=sampling_rate

    mode=mono/stereo

    

    '''

    max_ind = len(raw_data) 

    start_ind = min(int(start * sr), max_ind)

    end_ind = min(int(end * sr), max_ind)

    return raw_data[start_ind: end_ind]
sns.scatterplot(x=(data.end-data.start), y=data.pid)
sns.boxplot(y=(data.end-data.start))
os.makedirs('processed_audio_files')
for index,row in data.iterrows():

    print("Index ->",index)

    print("Data->\n",row)

    break
import librosa as lb

import soundfile as sf

i,c=0,0

for index,row in data.iterrows():

    maxLen=6

    start=row['start']

    end=row['end']

    filename=row['filename']

    

    #If len > maxLen , change it to maxLen

    if end-start>maxLen:

        end=start+maxLen

    

    audio_file_loc=path + filename + '.wav'

    

    if index > 0:

        #check if more cycles exits for same patient if so then add i to change filename

        if data.iloc[index-1]['filename']==filename:

            i+=1

        else:

            i=0

    filename= filename + '_' + str(i) + '.wav'

    

    save_path='processed_audio_files/' + filename

    c+=1

    

    audioArr,sampleRate=lb.load(audio_file_loc)

    pureSample=getPureSample(audioArr,start,end,sampleRate)

    

    #pad audio if pureSample len < max_len

    reqLen=6*sampleRate

    padded_data = lb.util.pad_center(pureSample, reqLen)

    

    sf.write(file=save_path,data=padded_data,samplerate=sampleRate)

print('Total Files Processed: ',c)