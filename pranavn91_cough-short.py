import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import scipy

import sklearn

import hmmlearn

import simplejson

!pip install pydub

!pip install eyed3

import eyed3

import pydub

import os

print(os.listdir("../input/latestwav/latestwav/"))

os.chdir("../input/latestwav/latestwav/")

os.getcwd()
#convert to wav

#Number of channels 2

#Sample width 2

#Frame rate. 44100

#Number of frames 873792

#parameters: _wave_params(nchannels=2, sampwidth=2, framerate=44100, nframes=873792, comptype='NONE', compname='not compressed')



from os import path

import wave, struct, math, random

from pydub import AudioSegment





# files                                                                         

audio_files = os.listdir()

obj = wave.open(audio_files[2], mode='rb')

print( "Number of channels",obj.getnchannels())

print ( "Sample width",obj.getsampwidth())

print ( "Frame rate.",obj.getframerate())

print ("Number of frames",obj.getnframes())

print ( "parameters:",obj.getparams())
!pip install pyAudioAnalysis

import pyAudioAnalysis



from pyAudioAnalysis import audioBasicIO

from pyAudioAnalysis import ShortTermFeatures

import matplotlib.pyplot as plt



#dont run this

#import matplotlib.pyplot as plt

#from pyAudioAnalysis import audioBasicIO

#from pyAudioAnalysis import audioFeatureExtraction



#https://medium.com/heuristics/audio-signal-feature-extraction-and-clustering-935319d2225



def preProcess( fileName ):

    [Fs, x] = audioBasicIO.read_audio_file(fileName) #A



    #B

    if( len( x.shape ) > 1 and  x.shape[1] == 2 ):

        x = np.mean( x, axis = 1, keepdims = True )

    else:

        x = x.reshape( x.shape[0], 1 )

    #C

    F, f_names = ShortTermFeatures.feature_extraction(

        x[ :, 0 ],

        Fs, 0.050*Fs,

        0.025*Fs

    )

    

    return (f_names, F)
import os

os.chdir("/kaggle/input/editedcoughs/outdir - Copy")

print(os.listdir("/kaggle/input/editedcoughs/outdir - Copy"))

audio = pd.read_csv("audioonly.csv")
df = audio.drop('Unnamed: 0', axis=1)

df = df.drop('filename', axis=1)

df = df.drop('output', axis=1)
audio_exploded = pd.concat([pd.DataFrame(df[col].str.split().values.tolist())\

                              .add_prefix(f'{col}_')

                            for col in df.columns],

                            axis=1)
audio_final = pd.concat([audio_exploded.reset_index(drop=True),audio['filename'],audio['output']], axis=1)
audio_final = audio_final.replace(',|\[|\]', '', regex=True)
audio_final.to_csv("audio_final.csv", sep=',', encoding='utf-8')