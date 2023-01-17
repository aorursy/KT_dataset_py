import matplotlib.pyplot as plt

import math

import numpy as np

import random



def reorder(x):

    return np.hstack([x[(x.shape[-1]+1)//2:], x[:(x.shape[-1]+1)//2]])



def get_data():

    f = open("../input/f8.txt","r")

    text = f.read()

    items = text.split(" ")

    items = [float(item) for item in items]

    return items



data = get_data()

time = np.fft.fftfreq(len(data),0.01)

fourier = reorder(np.abs(np.fft.fft(data))/len(data)*2)

#fourier = [abs(item) for item in fourier]

plt.plot(reorder(time),fourier)

deltf = 1/5

maxs = []

for k in range(len(fourier)):

    if k == 0:

        if fourier[k]>fourier[k+1]:

            maxs.append(fourier[k]*deltf)

    elif k == len(fourier)-1:

        if fourier[k]>fourier[k-1]:

            maxs.append(fourier[k]*deltf)

    else:

        if fourier[k]>fourier[k+1] and fourier[k]>fourier[k-1]:

            maxs.append(fourier[k]*deltf)

maxs