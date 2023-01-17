import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import math

import gc

import warnings

import os

from glob import glob

from PIL import Image

import cv2

import pydicom

from IPython.display import display, Audio

import folium



from sklearn.preprocessing import minmax_scale



import librosa

from librosa.display import waveplot, specshow

from librosa.feature import melspectrogram, chroma_cqt, mfcc, delta, spectral_bandwidth, spectral_centroid

from librosa.beat import beat_track



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)



%matplotlib inline
MAIN_PATH = '../input/birdsong-recognition/'



TRAIN_PATH = os.path.join(MAIN_PATH, 'train.csv')

TEST_PATH = os.path.join(MAIN_PATH, 'test.csv')

SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')



EXAM_TEST = os.path.join(MAIN_PATH, 'example_test_audio')

TRAIN_AUDIO_PATH = os.path.join(MAIN_PATH, 'train_audio')



SR = 32000
def normalize(x, axis=0):

    return minmax_scale(x, axis=axis)







def display_feature(df, feature, top=10):

    

    plt.figure(figsize=(15,8))

    sns.set_style('darkgrid')

    ax = sns.countplot(y=feature, data=df, order=df[feature].value_counts().index[:top])



    for p in ax.patches:

        ax.annotate('{:.2f}%'.format(100*p.get_width()/df.shape[0]), (p.get_x() + p.get_width() + 0.02, p.get_y() + p.get_height()/2))



    plt.title(f'Distribution of {feature}', size=25, color='b')

    plt.show()

    

    

    

def audio_map(df, audio_list, num_bird=5):

    for i in range(num_bird):

        rand_bird = random.choice(df['ebird_code'])

        rand_df = df[df['ebird_code']==rand_bird].sample().reset_index(drop=True)

        latitude = rand_df.loc[0, 'latitude']

        longitude = rand_df.loc[0, 'longitude']

        location = rand_df.loc[0, 'location']

        

        map_hooray = folium.Map(location=[latitude, longitude], zoom_start=10)

        folium.Marker([latitude, longitude], popup=location).add_to(map_hooray)

        

        file = random.choice(rand_df['filename'])

        print(f'{file}: {rand_bird} in{location}')

        file_path = [i for i in audio_list if file in i][0]

        display(Audio(data=file_path, autoplay=True))

        display(map_hooray)

        

        

        

def spectrum_wave(df, audio_list, num_bird=3):

    num_ax = 8

    for i in range(num_bird):

        rand_bird = random.choice(df['ebird_code'])

        rand_df = df[df['ebird_code']==rand_bird].sample().reset_index(drop=True)

        file = random.choice(rand_df['filename'])

        file_path = [i for i in audio_list if file in i][0]

        y, sr = librosa.load(file_path, sr=SR, offset=0, duration=10)

        

        S = melspectrogram(y, sr=sr, n_mels=128)

        log_s = librosa.power_to_db(S, ref=np.max)

        

        fig, ax = plt.subplots(num_ax, 1, figsize=(20, 7*num_ax))

        

        ax0 = ax[0].twinx()

        waveplot(y, sr, color='yellowgreen', alpha=0.4, ax=ax0)

        s0 = specshow(log_s, sr=sr, x_axis='time', y_axis='log', ax=ax[0])

        ax[0].set_title('Mel', color='r', fontsize=15)

        

        

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        s_harmonic   = melspectrogram(y_harmonic, sr=sr)

        s_percussive = melspectrogram(y_percussive, sr=sr)

        

        log_sh = librosa.power_to_db(s_harmonic, ref=np.max)

        log_sp = librosa.power_to_db(s_percussive, ref=np.max)

        

        ax1 = ax[1].twinx()

        waveplot(y, sr, color='yellowgreen', alpha=0.4, ax=ax1)

        specshow(log_sh, sr=sr, x_axis='time', y_axis='log', ax=ax[1])

        ax[1].set_title('Harmonic', color='r', fontsize=15)

        

        ax2 = ax[2].twinx()

        waveplot(y, sr, color='yellowgreen', alpha=0.4, ax=ax2)

        specshow(log_sp, sr=sr, x_axis='time', y_axis='log', ax=ax[2])

        ax[2].set_title('Percussive', color='r', fontsize=15)

        

        c = chroma_cqt(y_harmonic, sr=sr, bins_per_octave=36)

        

        specshow(c, sr=sr, x_axis='time', y_axis='log', ax=ax[3])

        ax[3].set_title('Chromagram', color='r', fontsize=15)

        

        tempo, beats = beat_track(y_percussive, sr=sr)

        

        specshow(log_s, sr=sr, x_axis='time', y_axis='log', ax=ax[4])

        ax[4].set_title('Beat track', color='r', fontsize=15)

        ax[4].vlines(librosa.frames_to_time(beats), 1, 0.5 * sr,

                     colors='w', linestyles='-', linewidth=2, alpha=0.5)

        

        mfcc_ = mfcc(S=log_s, n_mfcc=13)

        delta_mffc = delta(mfcc_)

        delta_mffc2 = delta(mfcc_, order=2)

        

        specshow(mfcc_, x_axis='time', y_axis='log', ax=ax[5])

        ax[5].set_title('MFFC', color='r', fontsize=15)

        

        specshow(delta_mffc, x_axis='time', y_axis='log', ax=ax[6])

        ax[6].set_title('Delta MFFC', color='r', fontsize=15)

        

        specshow(delta_mffc2, x_axis='time', y_axis='log', ax=ax[7])

        ax[7].set_title('Delta MFFC2', color='r', fontsize=15)

        

        plt.suptitle(f'{file}: {rand_bird}', fontsize=20, color='b')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])        

        

        

        

def bandwidth(df, audio_list, num_bird=5, num_each_bird=3):

    for i in range(num_bird):

        rand_bird = random.choice(df['ebird_code'])

        rand_df = df[df['ebird_code']==rand_bird].sample(num_each_bird).reset_index(drop=True)

        fig, ax = plt.subplots(num_each_bird, 1, figsize=(20, 10*num_each_bird))

        for row in range(len(rand_df)):

            file = rand_df.loc[row, 'filename']

            file_path = [i for i in audio_list if file in i][0]

            y, sr = librosa.load(file_path, sr=SR, offset=0, duration=10)

            spec_center = spectral_centroid(y, sr=sr)[0]

            band2 = spectral_bandwidth(y, sr=sr, p=2)[0]

            band3 = spectral_bandwidth(y, sr=sr, p=3)[0]

            band4 = spectral_bandwidth(y, sr=sr, p=4)[0]

#             spec_center = specshow(spec_center, sr=sr, x_axis='time', y_axis='log')



            waveplot(y, sr, color='b', alpha=0.4, ax=ax[row])

            ax0 = ax[row].plot(librosa.frames_to_time(range(len(spec_center))), normalize(spec_center), color='r')[0]

            ax1 = ax[row].plot(librosa.frames_to_time(range(len(band2))), normalize(band2), color='g')[0]

            ax2 = ax[row].plot(librosa.frames_to_time(range(len(band3))), normalize(band3), color='black')[0]

            ax3 = ax[row].plot(librosa.frames_to_time(range(len(band4))), normalize(band4), color='y')[0]

            fig.legend([ax0, ax1, ax2, ax3], ['center', 'band2', 'band3', 'band4'], fontsize=16)

            ax[row].set_title(f'{rand_df.loc[row, "filename"]}', color='r', fontsize=15)

            

            ax[row].legend()

                     

        plt.suptitle(f'{rand_bird}', fontsize=20, color='b')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])           

        plt.show()        
train_df = pd.read_csv(TRAIN_PATH, usecols=['ebird_code', 'recordist', 'location', 'file_type',

                                            'date', 'filename', 'url', 'longitude', 'latitude'])

train_df.head(10)
test_df = pd.read_csv(TEST_PATH)

test_df.tail()
sub_df = pd.read_csv(SUB_PATH)

sub_df.head()
train_mp3 = glob(f'{TRAIN_AUDIO_PATH}/*/*.*')

print(f'Number of file: {len(train_mp3)}')
display_feature(train_df, 'recordist', top=10)
audio_map(train_df, train_mp3, num_bird=3)
%%time



spectrum_wave(train_df, train_mp3)
bandwidth(train_df, train_mp3)