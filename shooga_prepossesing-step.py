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
!pip install window_slider
#PD file

import librosa

import librosa.display

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np

from window_slider import Slider





x, sr = librosa.load('/kaggle/input/cutandcleaned/cc-m4a-0bed6b67-c61b-4457-9bcd-2b6ae8a3a3e86555311007072116034.m4a')



print('filename:',filename)

print('length =',len(x)/sr,'sec')



#plt.figure(figsize=(14,4.5))

librosa.display.waveplot(x, sr=sr)

plt.show()



hop_length = 256

S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)

logS = librosa.power_to_db(abs(S))



#Plot Mel-spectrogram (original)

#plt.figure(figsize=(14,4.5))

librosa.display.specshow(logS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB')

plt.show()



list = []

bucket_size = 86

overlap_count = 43

slider = Slider(bucket_size,overlap_count)

slider.fit(logS)  

    

while True:

    window_data = slider.slide()

    list.append(window_data)

    print("size of dimension:",window_data.shape)

    #plt.figure(figsize=(14,4.5))

    librosa.display.specshow(window_data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+2.0f dB')

    plt.show()

    # do your stuff  

    if slider.reached_end_of_list(): break
print("number of window sliding:",len(list)) 

print("data is \n",window_data)
import os

path = "/kaggle/input/cutandcleaned/"

os.chdir(path)
import librosa

import librosa.display

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np

directory = "/kaggle/input/cutandcleaned/"

for filename in os.listdir(directory):

    if filename.endswith(".m4a"):

     #read audio file

        x, sr = librosa.load(filename)

        print('filename:',filename)

        print('length =',len(x)/sr,'sec')

        

        #Waveplot

        librosa.display.waveplot(x, sr=sr)

        plt.show()  



        

        #Mel-spectrogram

        hop_length = 256

        S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)

        logS = librosa.power_to_db(abs(S))



        #Plot Mel-spectrogram (original)

        librosa.display.specshow(logS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

        plt.colorbar(format='%+2.0f dB')

        plt.show()



        #defined windowsize and overlap

        window_size =86

        overlap = 43

        list = []



        #window sliding

        slider = Slider(window_size,overlap)

        slider.fit(logS) 

        while True:

          window_data = slider.slide()

          list.append(window_data)

          print("size of dimension:",window_data.shape)

          #do your stuff

          #print(window_data)

          #show mel-spectrogram

          librosa.display.specshow(window_data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

          plt.colorbar(format='%+2.0f dB')

          plt.show()

          if slider.reached_end_of_list(): 

            break



    else:

        continue