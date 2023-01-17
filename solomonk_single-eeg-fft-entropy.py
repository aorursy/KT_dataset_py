%reset -f

__author__ = 'Solomonk: https://www.kaggle.com/solomonk/'



import scipy.io

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import numpy

import pandas as pd

from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve



import numpy

import pandas

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from itertools import combinations



from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.datasets import dump_svmlight_file



import os

import numpy as np, h5py 

from scipy.io import loadmat

#--------------------DEFINE DATA IEEG SETS-----------------------#

DATA_FOLDER= '../input/train_1/'



#SINGLE MAT FILE FOR EXPLORATION

TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE=DATA_FOLDER + "/1_101_1.mat"

#--------------------DEFINE DATA SETS-----------------------#
from PIL import Image

import numpy as np



#---------------------------------------------------------------#

def entropy(signal):

    '''

    function returns entropy of a signal

    signal must be a 1-D numpy array

    '''

    lensig=signal.size

    symset=list(set(signal))

    numsym=len(symset)

    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]

    ent=np.sum([p*np.log2(1.0/p) for p in propab])

    return ent

#---------------------------------------------------------------#    



#---------------------------------------------------------------#

def ieegMatToPandasDF(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   



def ieegMatToArray(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return ndata['data']  



#---------------------------------------------------------------#



#---------------------------------------------------------------#

def ieegSingleMetaData(path):

    mat_data = scipy.io.loadmat(path)

    data = mat_data['dataStruct']

    for i in [data, data[0], data[0][0][0], data[0][0][0][0]]:

        print((i.shape, i.size))

#---------------------------------------------------------------#        



#---------------------------------------------------------------#

def ieegGetFilePaths(directory, extension='.mat'):

    filenames = sorted(os.listdir(directory))

    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]

    return files_with_extension

#---------------------------------------------------------------#



#---------------------------------------------------------------#

# EEG clips labeled "Preictal" (k=1) for pre-seizure data segments, 

# or "Interictal" (k-0) for non-seizure data segments.

# I_J_K.mat - the Jth training data segment corresponding to the Kth 

# class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).

def ieegIsInterictal(name):  

    try:

        return float(name[-5])

    except:

        return 0.0

#---------------------------------------------------------------#

ieegSingleMetaData(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)     
#1_121_1.mat 

x=ieegMatToPandasDF(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

print((x.shape, x.size))

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)

n=16

for i in range(0, n):

#     print i

    plt.subplot(n, 1, i + 1)

    plt.plot(x[i +1])
# Entropy

x=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

x_ent=x.ravel()

entropy(x_ent)



freq = np.fft.fft2(x)

freq = np.abs(freq)

print (freq)

freq.shape

z=np.log(freq).ravel()

z.shape

#type(x)

#x.head(3)
%matplotlib inline

x_std = x.std(axis=1)

# print(x_std.shape, x_std.ndim)

x_split = np.array(np.split(x_std, 100))

# print(x_split.shape)

x_mean = np.mean(x_split, axis=0)

# print(x_mean.shape)



plt.subplot(3, 1, 1)

plt.plot(x)

plt.subplot(3, 1, 2)

plt.plot(x_std)

plt.subplot(3, 1, 3)

plt.plot(x_mean)