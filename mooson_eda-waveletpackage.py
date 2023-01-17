from scipy.stats import kurtosis

import matplotlib.pyplot as plt

import scipy.signal as sg

from scipy import linalg

import seaborn as sns

import pandas as pd

import numpy as np

import graphviz

import warnings

import random

import time

import glob

import sys

import os
%time train = pd.read_csv('../input/train.csv',nrows=4000000,dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float64} ) # float32 is enough :)
train.shape[0]
train.head(5)
import pywt
wtp_data=train.acoustic_data

wtp_data=wtp_data-wtp_data.mean()

maxlevel=3

wavelet='db4'

wp=pywt.WaveletPacket(data=wtp_data.values.reshape(len(wtp_data)),wavelet=wavelet,mode='symmetric',maxlevel=maxlevel)

nodename=[node.path for node in wp.get_level(maxlevel, 'natural')]

fig,ax=plt.subplots(int(np.power(2,maxlevel)/2),2,figsize=(16,int(4*(np.power(2,maxlevel)/2))))

j=0

k=0

for i in nodename:

    ax[k][j].plot(wp[i].data)

    ax[k][j].set_title(i)

    j+=1

    if j>1:

        k+=1

        j=0

del wtp_data,wp