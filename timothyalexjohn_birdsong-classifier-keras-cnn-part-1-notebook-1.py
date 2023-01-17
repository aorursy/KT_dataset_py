import librosa         # Audio Manipulation Library
import librosa.display
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
train_dir = '../input/birdsong-recognition/train_audio/'
train_csv_dir = '../input/birdsong-recognition/train.csv'

test_dir = '../input/birdsong-recognition/test_audio/'
test_csv_dir = '../input/birdsong-recognition/test.csv'

train_df = pd.read_csv(train_csv_dir)
test_df = pd.read_csv(test_csv_dir)
train_filesave_dir = '/kaggle/working/train/' 
test_filesave_dir = '/kaggle/working/test/'

def create_spectrogram(df, start_time, duration, mode):
    
    if mode== 'train':
        filepath = (train_dir +df[2] +'/' +df[7])

    if mode== 'test':
        filepath = (test_dir +df[3] +'.mp3') 
        
    if mode== 'exa_test':                  
        file = '_'.join(df[0].split('_')[:-1])
        if file=='BLKFR-10-CPL_20190611_093000':
            filepath = (exa_test_dir +file +'.pt540.mp3')
        if file=='ORANGE-7-CAP_20190606_093000':
            filepath = (exa_test_dir +file +'.pt623.mp3')
    
    try:
        fig = plt.figure(figsize=[2.7,2.7])
        filename = filepath.split('/')[-1].split('.')[0]
        clip, sample_rate = librosa.load(filepath, sr=None, offset=start_time, duration=duration)
        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        if mode== 'train':
            if not os.path.exists(train_filesave_dir +df[2]):
                os.makedirs(train_filesave_dir +df[2] +'/')
            plt.savefig((train_filesave_dir + df[2] +'/' +filename +'.jpg'), bbox_inches='tight',pad_inches=0, facecolor='black')
        if mode== 'test':
            if not os.path.exists(test_filesave_dir):
                os.makedirs(test_filesave_dir) 
            plt.savefig((test_filesave_dir +filename +'.jpg'), bbox_inches='tight',pad_inches=0, facecolor='black')
        if mode== 'exa_test':
            if not os.path.exists(exa_test_filesave_dir):
                os.makedirs(exa_test_filesave_dir)
            plt.savefig((exa_test_filesave_dir + df[0] +'.jpg'), bbox_inches='tight',pad_inches=0, facecolor='black')
            
        fig.clear()          
        plt.close(fig)
        plt.close()
        plt.close('all')     # These Lines are Very Important!! If not given, Server will run out of allocated Memory
        plt.cla()
        fig.clf()
        plt.clf()
        plt.close()
    
    except:
        print("found a broken Audio File")      
# Getting Rid of Un-Wanted Warnings when Loading Audio Files
import warnings

warnings.filterwarnings("ignore")
start_time= 0

for row in train_df.values:
    create_spectrogram(row, start_time, row[6], mode= 'train')

for row in test_df.values:
    if row[0]=='site_3' :
        create_spectrogram(row, start_time, duration= None, mode= 'test')
    else :
        start_time = row[2] - 5
        create_spectrogram(row, start_time, duration= 5, mode= 'test')
# Trying on example_test_audio

exa_test_dir = '../input/birdsong-recognition/example_test_audio/'
exa_test_csv_dir = '../input/birdsong-recognition/example_test_audio_summary.csv'
exa_test_df = pd.read_csv(exa_test_csv_dir)

exa_test_filesave_dir = '/kaggle/working/exa_test/' 
 
for row in exa_test_df.values:
    if pd.isna(row[1])==False :
        start_time = row[3] - 5
        create_spectrogram(row, start_time, duration= 5, mode= 'exa_test')
import shutil

shutil.make_archive('train_zipped', 'zip', '/kaggle/working/train')
shutil.make_archive('test_zipped', 'zip', '/kaggle/working/test')
shutil.make_archive('exa_test_zipped', 'zip', '/kaggle/working/exa_test')