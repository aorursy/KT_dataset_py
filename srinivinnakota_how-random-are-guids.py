import numpy as np # linear algebra

import matplotlib.pyplot as plt

import numpy as np



# read the input file

guids= np.loadtxt(open("../input/guids2.csv", "rb"), delimiter=",", skiprows=0, dtype='str').reshape(-1,1)

m= guids.shape[0] # get number of samples



# What we just finished reading are strings. We need numbers, so we are creating a new array

guids2= np.zeros((m,1), dtype=float)

for i in range(0,m-1):

    guids2[i,0]= int(guids[i,0], 16)    # radix 16, so hexadecimal input, now being converted to decimal



# plot a histogram    

plt.hist(guids2, bins=100, color='magenta'); # the semicolon prevents the outputting of the array 