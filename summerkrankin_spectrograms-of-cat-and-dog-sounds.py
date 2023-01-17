import numpy as np  # linear algebra

import pandas as pd  # CSV file



import matplotlib.pyplot as plt

%matplotlib inline

#import scipy.io.wavfile as sci_wav  # Open wav files

#from scipy.fftpack import fft

import librosa

import librosa.display

ROOT_DIR = '../input/cats_dogs/'
# this one uses labrosa (kaggle doesn't have it) and is much prettier



def plot_spec(ROOT_DIR, filename):

    """ Load an audio file, plot the raw wav form (amplitude vs. time)

    calculate the STFT and plot the spectrogram on the log freq scale, power is converted to dB (decibels)"""

        

    # load file . y= numpy array, sr=22050 

    y, sr = librosa.load(ROOT_DIR + filename)

    dataC_c= y/2.**15

    duration = dataC_c.shape[0]/sr

    samplepoints = float(dataC_c.shape[0])



    # We can represent sound by plotting the pressure values against time axis.

    #Create an array of sample point in 1-D



    time_arr = np.arange(0,samplepoints,1)

    time_arr = (time_arr/sr)*2

       

    fig = plt.figure(figsize=(20,5));

    #plt.subplot(1,2,1)

    plt.title("raw signal " + str(filename),fontsize=25)

    

    librosa.display.waveplot(y, sr=sr)

    

    plt.xlabel('Time (ms/2)',fontsize = 25)

    plt.ylabel('Amplitude (RMS)',fontsize = 25)



    # TO MAKE tick fonts bigger, We define a fake subplot that is in fact only the plot.  

    plot = fig.add_subplot(111)

    # We change the fontsize of minor ticks label 

    plot.tick_params(axis='both', which='major', labelsize=20)

    plot.tick_params(axis='both', which='minor', labelsize=20)

    

    

   

    #to display spectrogram for a ind. file

    

    fig= plt.figure(figsize=(20,10))

    #plt.subplot(1,2,2)

   

    # calc the STFT with a window = 1024 . 

    D = librosa.core.stft(y, n_fft=1024)

 

    librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')



    plt.title('Power spectrogram'+ filename, fontsize = 25)

    plt.colorbar(format='%+2.0f dB')

    tight_layout()                 

    plt.xlabel('Time (ms)', fontsize = 25)

    plt.ylabel('Frequency (Khz)', fontsize = 25)

   

    # to change the tick font size, We define a fake subplot that is in fact only the plot.  

    plot = fig.add_subplot(111)

     # We change the fontsize of minor ticks label 

    plot.tick_params(axis='both', which='major', labelsize=20)

    plot.tick_params(axis='both', which='minor', labelsize=20)
fft_spec('cat_78.wav')

plot_spec('./cats/george97.wav')
plot_spec('./cats/cat_90.wav')
plot_spec('./cats/cat_5.wav')
plot_spec('./dogs/dog_barking_4.wav')
plot_spec('./dogs/dog_barking_50.wav')
plot_spec('./dogs/dog_barking_94.wav')
plot_spec('./dogs/dog_barking_17.wav')
#file = './dogs/dog_barking_94.wav'

file = './cats/george97.wav'





y, sr = librosa.load(file)

# set the length of the array here if file is long

#y= y[:40000 ]

D = librosa.core.stft(y, n_fft=1024)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5));

librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')

#librosa.display.waveplot(y, sr=sr)

plt.title('Power spectrogram'+ file)

plt.colorbar(format='%+2.0f dB')

plt.tight_layout()