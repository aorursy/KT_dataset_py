from __future__ import print_function

import pandas as pd

import pickle

import numpy as np

import glob, os, re

import librosa



from pylab import *

from scipy.fftpack import fft



# And the display module for visualization

import librosa.display



import matplotlib.pyplot as plt

%matplotlib inline



from sqlalchemy import create_engine



path = '/your path for storing pickle files/'
def catdog_files(direct, form, animal):

    """ Generate a list of paths to the all the sound files in the directory, 

       

   IN:  direct = path to directory where sound files are stored (string)

         form = extention of the files (string) i.e. 'wav' 

                  

    OUT: save to a .csv containing the path, filename and ID for each .wav file"""

      

    filenames = librosa.util.find_files(direct, ext=[form])

    

    # grab all the .wav files in the directory

    os.chdir(direct)    

    filez = []    

    for file in glob.glob("*.wav"):

        filez.append(file)

    

    # get the number for each file for ID purposes store in file_id

    file_id = []

    for item in filez:

        id_ = re.findall('[0-9]+', item)

        id_ = str(id_)

        id_ = id_.strip("'[]'")

        file_id.append(id_)

    

    # save to df

    path_d = {'path':filenames}

    path_df = pd.DataFrame(data=path_d)

    

    id_d = {'id': file_id}

    id_df = pd.DataFrame(data=id_d)

    

    return path_df, id_df
dog_path_list, dog_id_list = catdog_files('./dogs/', 'wav')

% cd ../

cat_path_list, cat_id_list = catdog_files('./cats/', 'wav')

% cd ../
def load_catdog_wav(direct, filenames, file_id, animal):

    """ load .wav files, convert to floating point time series numpy array, 

    Calculate the 1D FFT/PSD for each and save in binary file

    -----------------------

   

   IN:  direct = path to directory where binary files will be saved

        filenames = a dataframe of paths to the all the sound files in the directory

        file_id = dataframe of the ID (number) associated with each sample 

        animal = animal that the file was recorded from 

         

    OUT: binary files for each sample

        fft = fft values (power)

        freqs = freqs for fft

        raw_ts = raw numpy array of wav file"""

          

    for i in range(len(filenames)):

        

        filename = filenames.iloc[i].values

        filename = str(filename)

        filename = filename.strip("'[]'")

        

        id_ = file_id.iloc[i].values

        id_ = str(id_)

        id_ = id_.strip("'[]'")

       

        #get raw time sereis and sample rate

        #sampling rate which will be the default of 22050 

        # y = audio as a np array

        y, sr = librosa.load(filename)

        

        #psd 

        ps = np.abs(np.fft.fft(y))**2



        ## Get frequencies corresponding to signal PSD

        time_step = 1 / sr

        

        freqs1 = np.fft.fftfreq(y.size, time_step)



        #only need the positive half

        index_pos = freqs1 > 0

        freqs = freqs1[index_pos]



        periodo = ps[index_pos]

        

        # take log10 for decibel units and scaling 

        periodogram = 10 * np.log10(periodo)

        

        # save as binary files      

        np.save( direct + animal + id_ + '_fft' , periodogram) 

        np.save( direct + animal + id_ + '_freqs' , freqs ) 

        np.save( direct + animal + id_ + '_raw_ts' , y ) 

load_catdog_wav('./cats/', cat_path_list, cat_id_list, 'cat')
load_catdog_wav('./dogs/', dog_path_list, dog_id_list, 'dog')
def resample_freq(direct):

    """ load master frequencies as template. get each file's fft and 

    find the nearest frequency to the one in template, 

    then get the index for that freq and save, repeat till you get to 818

   

   IN:  direct = path to directory where fft and freq files are stored (string)

         

    OUT: fft_df = dataframe containing the fft values for each sample(row) by freq(col) 

         freq_df =df with the corresponding freqs """

    

    def find_nearest_freq(array,value):

        indexx = (np.abs(array-value)).argmin()

        return indexx

    

    master_freq = np.load('master_frequencies.npy') 

    

    # get the lists of the 2 binary files we will need

    

    os.chdir(direct)

    file_freq= []

    for file in glob.glob("*_freqs.npy"):

        file_freq.append(file)

    

    file_fft= []

    for file in glob.glob("*_fft.npy"):

        file_fft.append(file)

    

    

    # create new dfs

    freq_df = pd.DataFrame(index=range(len(master_freq)),columns=range(len(file_fft)))

    fft_df = pd.DataFrame(index=range(len(master_freq)),columns=range(len(file_fft)))

  

    col = 0

    

    # open the fft and freqs for a sample

    for file1, file2 in zip(file_fft,file_freq):

        old_fft = np.load(file1)

        old_freq = np.load(file2)

        idf = []

    

        #for each freq in master, find the index for  the nearest freq

        for freq in master_freq:

        

            ind = find_nearest_freq(old_freq,freq)

            idf.append(ind)

            

        # using our index, pull out freqs, and fft (power) values 

        new_fft = old_fft[idf]

        new_freq = old_freq[idf]

        

        # make a new df for this animal

        fft_df[col] = pd.Series(new_fft)

        freq_df[col] = pd.Series(new_freq)

        

        #save as binary file

        np.save('nfft', file1  , new_fft ) 

        np.save('nfreq' , file1 , new_freq ) 



        col += 1

    

    return fft_df, freq_df
# open file w/ master freqs

master_freq = np.load('master_frequencies.npy') 

master_freq.shape
C_fft_df, C_freq_df = resample_freq('./cats/')
%cd ../
D_fft_df, D_freq_df = resample_freq('./dogs/')
dog = D_fft_df.T

dog['y_val'] = 1

cat = C_fft_df.T

cat['y_val'] = 0



# concat into one

both = pd.concat([cat,dog])
with open(path + 'both.pkl', 'wb') as picklefile:

        pickle.dump(both, picklefile)    

with open(path + 'dog.pkl', 'wb') as picklefile:

        pickle.dump(dog, picklefile)    

with open(path + 'cat.pkl', 'wb') as picklefile:

        pickle.dump(cat, picklefile)            