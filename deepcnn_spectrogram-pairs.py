import datetime

import pandas as pd

import numpy as np

from scipy.io import loadmat

from operator import itemgetter

import random

import os

import time

import glob

from matplotlib import pyplot as plt

from pylab import plot, show, subplot, specgram, imshow



random.seed(2075)

np.random.seed(2075)

Fs = 400

NFFT = 1024

N = 16; ch = 16

m = 80000 # This will give 200 x 200 spectrogram

    

def mat_to_dataframe(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])

    

def plot_spectrogram(pairs):



    pair = pairs[0]

    

    X0 = mat_to_dataframe(pair[0]).as_matrix()

    X1 = mat_to_dataframe(pair[1]).as_matrix()

    

    for i in range(N):         

        plt.subplot(2, 1, 1)

        plt.plot(X0[:5000,i])

        plt.title('ch ' + str(i) + ': preictal absent or class 0')

        

        plt.subplot(2, 1, 2)

        plt.plot(X1[:5000,i])

        plt.title('ch ' + str(i) + ': preictal present or class 1')

        

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.8)

        plt.show()

        

        plt.subplot(2, 1, 1)

        Pxx, freqs, bins, im = plt.specgram(X0[:m,i], NFFT=NFFT, Fs=Fs, noverlap=512, cmap=plt.cm.jet)

        plt.title('ch ' + str(i) + ': preictal absent or class 0')      

                

        plt.subplot(2, 1, 2)

        Pxx, freqs, bins, im = plt.specgram(X1[:m,i], NFFT=NFFT, Fs=Fs, noverlap=512, cmap=plt.cm.jet)

        plt.title('ch ' + str(i) + ': preictal present or class 1')

        

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.8)

        cax = plt.axes([0.85, 0.1, 0.075, 0.8])

        plt.colorbar(cax=cax)

        plt.show()
pairs = []

start = 11; stop = 15

for i in range(start,stop):

    pairs.append(['../input/train_1/1_' + str(i) + '_0.mat', 

                  '../input/train_1/1_' + str(i) + '_1.mat'])



plot_spectrogram(pairs)