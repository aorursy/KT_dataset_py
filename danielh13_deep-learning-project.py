# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import librosa

directory = '/kaggle/input/freesound-audio-tagging/audio_train/'

time_series = []

sampling_rate = []

tempos = []

all_beat_frames = []

all_beat_times = []



for dirname, _, filenames in os.walk(directory):

    print(len(filenames))

    print(len(dirname))





    for filename in filenames:

        print(len(filenames))



        y, sr = librosa.load(directory + filename)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        

        # SHANE IS GAY

        #print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        time_series.append(y)

        sampling_rate.append(sr)

        tempos.append(tempo)

        all_beat_frames.append(beat_frames)

        all_beat_times.append(beat_times)

# Any results you write to the current directory are saved as output.



df = pd.DataFrame([time_series,sampling_rate,tempos,all_beat_frames,all_beat_times],columns=['time_series', 'sampling_rate', 'tempo','beat_frames','beat_times'])



print('Ayyyyyyy CORONA')