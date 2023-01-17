# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/lab1_tools.py", dst = "../working/lab1_tools.py")

copyfile(src = "../input/lab1_proto.py", dst = "../working/lab1_proto.py")



# import all our functions

#from my_functions import *
import lab1_tools

import lab1_proto
# DT2119, Lab 1 Feature Extraction



# Function given by the exercise ----------------------------------



import numpy as np

import scipy.signal as signal

import scipy.fftpack as fftpack

import matplotlib.pyplot as plt

from lab1_tools import lifter





def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, 

         nceps=13, samplingrate=20000, liftercoeff=22):

    """Computes Mel Frequency Cepstrum Coefficients.



    Args:

        samples: array of speech samples with shape (N,)

        winlen: lenght of the analysis window

        winshift: number of samples to shift the analysis window at every time step

        preempcoeff: pre-emphasis coefficient

        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)

        nceps: number of cepstrum coefficients to compute

        samplingrate: sampling rate of the original signal

        liftercoeff: liftering coefficient used to equalise scale of MFCCs



    Returns:

        N x nceps array with lifetered MFCC coefficients

    """

    mspec_ = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)

    ceps = cepstrum(mspec_, nceps)

    return lifter(ceps, liftercoeff)





def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):

    """Computes Mel Filterbank features.



    Args:

        samples: array of speech samples with shape (N,)

        winlen: lenght of the analysis window

        winshift: number of samples to shift the analysis window at every time step

        preempcoeff: pre-emphasis coefficient

        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)

        samplingrate: sampling rate of the original signal



    Returns:

        N x nfilters array with mel filterbank features (see trfbank for nfilters)

    """

    frames = enframe(samples, winlen, winshift)

    preemph = preemp(frames, preempcoeff)

    windowed = windowing(preemph)

    spec = powerSpectrum(windowed, nfft)

    return logMelSpectrum(spec, samplingrate)



# Functions to be implemented ----------------------------------



def enframe(samples, winlen, winshift):

    """

    Slices the input samples into overlapping windows.



    Args:

        winlen: window length in samples.

        winshift: shift of consecutive windows in samples

    Returns:

        numpy array [N x winlen], where N is the number of windows that fit

        in the input signal

    """

    x = samples.shape[0] #

    print(x)

    y = x // winlen * 2 -1

    segment = np.zeros((y,winlen))

    for i in range(y):

        segment[i,:] = samples[i*winshift:i*winshift+winlen]

    #segment[y,:x-y*winshift-1] = samples[y*winshift:x-1]

    

    return segment





example = np.load('../input/lab1_example.npz')['example'].item()

segment = enframe(example['samples'],400,200)

plt.figure()

plt.plot(example['samples'])

plt.title('original signal')

plt.show()



plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(segment)

plt.title('segment samples')

plt.subplot(2,1,2)

plt.pcolormesh(example['frames'])

plt.title('example')

#print(segment.shape,example['samples'].shape)
def preemp(input, p=0.97):

    """

    Pre-emphasis filter.



    Args:

        input: array of speech frames [N x M] where N is the number of frames and

               M the samples per frame

        p: preemhasis factor (defaults to the value specified in the exercise)



    Output:

        output: array of pre-emphasised speech samples

    Note (you can use the function lfilter from scipy.signal)

    """

    b = np.array([1,-p])

    a = 1

    N = input.shape[0]

    M = input.shape[1]

    x = np.zeros((N,M))

    for i in range(N):

        x[i,:] = signal.lfilter(b,a,input[i,:])

    return x



preemp_ = preemp(segment)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(preemp_)

plt.title('preemph')

plt.subplot(2,1,2)

plt.pcolormesh(example['preemph'])

plt.title('example')

#print(preemp.shape, example['preemph'].shape)

def windowing(input):

    """

    Applies hamming window to the input frames.



    Args:

        input: array of speech samples [N x M] where N is the number of frames and

               M the samples per frame

    Output:

        array of windoed speech samples [N x M]

    Note (you can use the function hamming from scipy.signal, include the sym=0 option

    if you want to get the same results as in the example)

    """

    N = input.shape[0]

    M = input.shape[1]

    x = np.zeros((N,M))

    window = signal.hamming(M,sym=False)

    plt.figure()

    xx = np.linspace(0,399,400)

    plt.bar(xx,window)

    plt.title("hamming window")

    for i in range(N):

        x[i,:] = input[i,:]*window

        

    return x





windowed = windowing(preemp_)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(windowed)

plt.title('windowed')

plt.subplot(2,1,2)

plt.pcolormesh(example['windowed'])

plt.title('example')

#print(windowed.shape,example['windowed'].shape )
def powerSpectrum(input, nfft):

    """

    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT



    Args:

        input: array of speech samples [N x M] where N is the number of frames and

               M the samples per frame

        nfft: length of the FFT

    Output:

        array of power spectra [N x nfft]

    Note: you can use the function fft from scipy.fftpack

    """

    N = len(input)

    M = nfft

    x = np.zeros((N,M))

    x = fftpack.fft(input,M)

    x = abs(x)

    x = x*x

    

    return x



spec = powerSpectrum(windowed, 512)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(spec)

plt.title('spec')

plt.subplot(2,1,2)

plt.pcolormesh(example['spec'])

plt.title('example')

#print(spec.shape,example['spec'].shape)

diff = spec - example['spec']
import lab1_tools



def logMelSpectrum(input, samplingrate):

    """

    Calculates the log output of a Mel filterbank when the input is the power spectrum



    Args:

        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and

               nfft the length of each spectrum

        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)

    Output:

        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number

        of filters in the filterbank

    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and

          nmelfilters

    """

    N = len(input)#92frames

    Mel = lab1_tools.trfbank(samplingrate,len(input[0]))#40filters*512

    M = Mel.shape[0]

    # plot the filters in linear frequency scale x=k*512/fs(20000)

    plt.figure()

    for m in range(int(M/4)):

        plt.plot(Mel[m*4], label = m*4+1)

    plt.legend()

    plt.title('Mel filters in linear frequency')

    #plt.figure()

    #plt.plot(Mel[1])

    #plt.xlim(0,20)

    

    x = np.zeros((N,M))

    for i in range(N):

        for j in range(M):

            x[i,j] = np.log(np.sum(input[i,:]*Mel[j,:]))

        

    return x



mspec = logMelSpectrum(spec, 20000)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(mspec)

plt.title('mspec')

plt.subplot(2,1,2)

plt.pcolormesh(example['mspec'])

plt.title('example')
def cepstrum(input, nceps):

    """

    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform



    Args:

        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the

               number of frames and nmelfilters the length of the filterbank

        nceps: number of output cepstral coefficients

    Output:

        array of Cepstral coefficients [N x nceps]

    Note: you can use the function dct from scipy.fftpack.realtransforms

    """

    N = len(input)

    M = nceps

    x = np.zeros((N,M))

    

    y = fftpack.dct(input, type=2, norm='ortho') # 91*40

    #y = fftpack.dct(input, type=2, n=nceps) #y=91*13  n= length of transform

    x = y[:, :13]

    lx = lifter(x)

        

    return x



mfcc_ = cepstrum(mspec, 13)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(mfcc_)

plt.title('mfcc')

plt.subplot(2,1,2)

plt.pcolormesh(example['mfcc'])

plt.title('example')



'''

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(lmfcc_)

plt.title('lmfcc')

plt.subplot(2,1,2)

plt.pcolormesh(example['lmfcc'])

plt.title('example')

'''
# test mfcc function

import matplotlib.pyplot as plt

example = np.load('../input/lab1_example.npz')['example'].item()

test = example['samples']

res = lab1_proto.mfcc(test)

plt.figure()

plt.subplot(2,1,1)

plt.pcolormesh(res)

plt.title('lmfcc')

plt.subplot(2,1,2)

plt.pcolormesh(example['lmfcc'])

plt.title('example')
data = np.load('../input/lab1_data.npz')['data']
def correlation(data):

    '''

    Input:

    

    [N*M]array of features, N is the total number of frames in the dataset,

    M is the number of coefficient

    Output: M * M correlation matrix between feature coefficients

    '''

    

    result1 = lab1_proto.mfcc(data[0]['samples'])

    result2 = lab1_proto.mspec(data[0]['samples'])

    features = []

    features.append(result1)

    for i in range(data.shape[0]):

        sample = data[i]['samples']

        sp = lab1_proto.mfcc(sample)#n*M

        result1 = np.concatenate((result1,sp),axis=0)

        features.append(sp.T)

        sp2 = lab1_proto.mspec(sample)

        result2 = np.concatenate((result2,sp2),axis=0)

    #print(result1.shape)#(3939*13)

    corr1 = np.corrcoef(result1.T)

    corr2 = np.corrcoef(result2.T)

    plt.figure()

    plt.pcolormesh(corr1)

    plt.title('Correlation between MFCC features')

    plt.figure()

    plt.pcolormesh(corr2)

    plt.title('Correlation between mspec features')

    

    return features, result1

    

features, result1 = correlation(data)
# Explore Speech Segments with Clustering

import sklearn.mixture as mixture

from scipy import linalg

import itertools

import matplotlib as mpl

n_components = [4]#,8,16,32]

eg = [1, 2, 23, 24]



color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',

                              'darkorange'])



for n in n_components:

    gmm = mixture.GaussianMixture(n_components=n)

    Y_ = gmm.fit(result1)

    posteriors = gmm.predict_proba(result1)#n_sample * n

    plt.figure()

    plt.title('{} components'.format(n))

    plt.pcolormesh(posteriors)

    for i,idx in enumerate(eg):

        sample = data[idx]['samples']

        f = lab1_proto.mfcc(sample)#n*M

        ax1 = plt.subplot(4,2,i+1)

        ax1.pcolormesh(f)

        g_post = gmm.predict_proba(f)

        ax2 = plt.subplot(4,2,(i+1)+4)

        #ax2.pcolormesh(g_post)

        #ax2.plot(g_post)

        for k in range(n):

            x = np.linspace(0,g_post.shape[0],g_post.shape[0])

            ax2.fill(x, g_post[:,k], alpha=0.5)

        #print(g_post[:,3])
gmm = mixture.GaussianMixture(n_components=4).fit(result1)

sample = data[1]['samples']

f = lab1_proto.mfcc(sample)

g_post = gmm.predict_proba(f)

plt.pcolormesh(g_post)
sample = data[2]['samples']

f = lab1_proto.mfcc(sample)

g_post = gmm.predict_proba(f)

plt.pcolormesh(g_post)
sample = data[23]['samples']

f = lab1_proto.mfcc(sample)

g_post = gmm.predict_proba(f)

plt.pcolormesh(g_post)
sample = data[24]['samples']

f = lab1_proto.mfcc(sample)

g_post = gmm.predict_proba(f)

plt.pcolormesh(g_post)

sample.shape

f.shape
gmm = mixture.GaussianMixture(n_components=32).fit(result1)

sample = data[39]['samples']

f = lab1_proto.mfcc(sample)

g_post = gmm.predict_proba(f)

plt.pcolormesh(g_post)
def dtw(x, y, dist):

    """Dynamic Time Warping.



    Args:

        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality

              and N, M are the respective lenghts of the sequences

        dist: distance function (can be used in the code as dist(x[i], y[j]))



    Outputs:

        d: global distance between the sequences (scalar) normalized to len(x)+len(y)

        LD: local distance between frames from x and y (NxM matrix)

        AD: accumulated distance between frames of x and y (NxM matrix)

        path: best path thtough AD



    Note that you only need to define the first output for this exercise.

    """

    N = x.shape[0]

    M = y.shape[0]

    LD = np.zeros((N,M))

    AD = np.zeros((N,M))

    for a in range(N):

        for b in range(M):

            LD[a,b] = dist(x[a,:],y[b,:])

    AD[0,0] = LD[0,0]

    for a in range(1,N):

        AD[a,0] = LD[a,0] + AD[a-1,0]

    for b in range(1,M):

        AD[0,b] = LD[0,b] + AD[0,b-1]

    for a in range(1,N):

        for b in range(1,M):

            AD[a,b] = LD[a,b] + min(AD[a-1,b],AD[a-1,b-1],AD[a,b-1])



    b = AD[N-1,M-1] / (M+N)

    return b

def dist(x,y):

    import math

    N = x.shape[0]

    M = y.shape[0]

    distance = 0

    for i in range(N):

        distance = distance + (x[i]-y[i])*(x[i]-y[i])

    distance = math.sqrt(distance)

    return distance

sample_1 = data[17]['samples']

mfcc_1 = lab1_proto.mfcc(sample_1)

sample_2 = data[16]['samples']

mfcc_2 = lab1_proto.mfcc(sample_2)

distance = dtw(mfcc_1,mfcc_2,dist)

print(distance)
data.shape
D = data.shape[0]

GD = np.zeros((D,D))

for i in range(D):

    for k in range(D):

        sample_1 = data[i]['samples']

        sample_2 = data[k]['samples']

        mfcc_1 = lab1_proto.mfcc(sample_1)

        mfcc_2 = lab1_proto.mfcc(sample_2)

        GD[i,k] = dtw(mfcc_1,mfcc_2,dist)

plt.pcolormesh(GD)
plt.pcolormesh(GD)
gd = np.zeros((4,4))

ind = [16,17,38,39]

for i,idx in enumerate(ind):

    for k,idc in enumerate(ind):

        gd[i,k] = GD[idx,idc]

plt.pcolormesh(gd)
from lab1_tools import tidigit2labels
label = tidigit2labels(data)
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(GD,'complete')

plt.figure(figsize = (30,10))

dn = dendrogram(Z,labels = label)

plt.show()