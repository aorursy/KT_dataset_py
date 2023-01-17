# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import cv2

import random

import scipy

from scipy import signal

from scipy.stats import rayleigh,erlang,uniform,exponweib,gamma
X = np.arange(-50,50)
X_Gaussian = signal.gaussian(100,std=7)

X_Rayleigh = rayleigh.pdf(X)

X_Uniform = uniform.pdf(X,-20,20)

X_Gamma = gamma.pdf(X,3,2)

X_Expomential = exponweib.pdf(X,10,1)
fig = plt.figure(figsize=(12, 40))

fig1 = fig.add_subplot(10, 3, 1)

fig2 = fig.add_subplot(10, 3, 2)

fig3 = fig.add_subplot(10, 3, 3)

fig4 = fig.add_subplot(10, 3, 4)

fig5 = fig.add_subplot(10, 3, 5)

fig6 = fig.add_subplot(10, 3, 6)



fig1.title.set_text('Origin')

fig1.plot(X)

fig2.title.set_text('Gaussian')

fig2.plot(X_Gaussian)

fig3.title.set_text('Rayleigh')

fig3.plot(X_Rayleigh)

fig4.title.set_text('Uniform')

fig4.plot(X_Uniform)

fig5.title.set_text('Gamma')

fig5.plot(X_Gamma)

fig6.title.set_text('Expomential')

fig6.plot(X_Expomential)



plt.show()
img = cv2.imread('/kaggle/input/cameraman.tif')/255

plt.imshow(img)

plt.show()
# from skimage import color

# row, col, ch = img.shape

# mean = 0

# var = 0.1

# sigma = var**0.5



# X_Gaussian = np.random.normal(mean,sigma,(row,col,ch)).reshape(row,col,ch)

# X_Rayleigh = rayleigh.pdf(img).reshape(row,col,ch)

# X_Uniform = np.random.uniform(img).reshape(row,col,ch)

# X_Gamma = np.random.gamma(img).reshape(row,col,ch)

# X_Expomential = np.random.exponential(img).reshape(row,col,ch)



# figs = []

# names = ['Origin', 'Gaussian', 'Rayleigh', 'Uniform', 'Gamma', 'Expomential']

# noises = [np.zeros(img.shape),X_Gaussian,X_Rayleigh,X_Uniform,X_Gamma,X_Expomential]

# fig = plt.figure(figsize=(12, 40))

    

# for (index,noise) in zip(range(6),noises):

#     figs.append(fig.add_subplot(10, 2, index*2+1))

#     figs[index*2].title.set_text(names[index])

#     figs[index*2].imshow(img + noise)

#     figs.append(fig.add_subplot(10, 2, index*2+2))

#     figs[index*2+1].title.set_text(names[index])



# plt.show()
from skimage import color

row, col, ch = img.shape

mean = 0

var = 0.1

sigma = var**0.5



X_Gaussian = np.random.normal(mean,sigma,(row,col,ch)).reshape(row,col,ch)

X_Rayleigh = rayleigh.pdf(img).reshape(row,col,ch)

X_Uniform = np.random.uniform(img).reshape(row,col,ch)

X_Gamma = np.random.gamma(img).reshape(row,col,ch)

X_Expomential = np.random.exponential(img).reshape(row,col,ch)



figs = []

names = ['Origin', 'Gaussian', 'Rayleigh', 'Uniform', 'Gamma', 'Expomential']

noises = [np.zeros(img.shape),X_Gaussian,X_Rayleigh,X_Uniform,X_Gamma,X_Expomential]

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise_img = img + noise[2]

plt.imshow(img)

plt.show()

plt.imshow(noise_img)

plt.show()
# median_img = cv2.medianBlur(np.float32(noise_img), 3)

# plt.imshow(median_img)

# plt.show()



for i in range(1,5,2):

    img1Fix = cv2.medianBlur(np.float32(noise_img),i)

    plt.imshow(img1Fix)

    plt.show()

    print("Applied filter %i" %i)
from scipy.signal import convolve2d

from skimage import color, data, restoration



def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



noise_gray_img = rgb2gray(noise_img)

plt.imshow(noise_gray_img)

plt.show()

print("noise imgae in gray")
psf = np.ones((5, 5)) / 25

new_img = convolve2d(noise_gray_img, psf, 'same')

plt.imshow(new_img)

plt.show()

new_img += 0.1 * new_img.std() * np.random.standard_normal(new_img.shape)

for i in range(100,1101,100):

    deconvolved_img = restoration.wiener(new_img, psf, i)

    plt.imshow(deconvolved_img)

    plt.show()