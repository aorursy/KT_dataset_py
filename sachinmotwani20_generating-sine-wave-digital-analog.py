#Signal Processing using Python 1 

''' https://www.youtube.com/watch?v=t3AEweWweSI&t=160s'''
from __future__ import division 

#Importing Python 3 division 

'''In Python2 Divsion:

        5/2=2  (integer)

    In Python3 Division:

        5/2=2.5 (float)'''

%matplotlib inline

#To ensure that No new window opens for output graph/image 

import numpy as np

import math 

import matplotlib.pyplot as plt

import sys
#Creating a sine wave 

freq=50

time_period=1/freq #20ms

time = time_period*2 #40ms (Time period)

amplitude=2
t=np.linspace(0, time, 500, endpoint=True)

 #(start, end, total number of points from start will be spaced, whether endpoint will be included or not)

x=2*math.pi*freq*t #sine angle

# y = asin(omega * t)

yc=amplitude*np.sin(x)
#Visualising the Signal

#To create an empty figure

fig1= plt.figure(figsize=(10,4)) #(length, height)

axes1=fig1.add_axes([0.1,0.1,0.8,0.8]) 

#Plotting

axes1.plot(x,yc, color='red', linewidth=3, linestyle='-')



axes1.set_ylim([-3,-3])

axes1.set_xlim([0,np.max(x)])



axes1.set_xticks((2*math.pi*freq)*(np.arange(0,41,5)*1e-3));

#Creating an array from 0 to 40 with steps of 5 

#Multiplying it with 10^-3 to convert into msec

axes1.set_xticklabels(np.arange(0,41,5), fontsize=14);



axes1.set_yticks([-3,-2,-1,0,1,2,3])

axes1.set_yticklabels([-3,-2,-1,0,1,2,3], fontsize=14);



axes1.set_xlabel("time(ms)", fontsize=18);

axes1.set_ylabel("Amplitude", fontsize=18);

axes1.set_title("Signal Vs time", fontsize=18)
#Adding Random noise to the Signal

yc=amplitude*np.sin(x)+ 1*np.random.randn(len(x))

#use 'rand' for Uniform Value

#use 'randn' for Gaussian
#Visualising the Signal with added random noise

#To create an empty figure

fig1= plt.figure(figsize=(10,4)) #(length, height)

axes1=fig1.add_axes([0.1,0.1,0.8,0.8]) 

#Plotting

axes1.plot(x,yc, color='red', linewidth=3, linestyle='-')



axes1.set_ylim([-3,-3])

axes1.set_xlim([0,np.max(x)])



axes1.set_xticks((2*math.pi*freq)*(np.arange(0,41,5)*1e-3));

#Creating an array from 0 to 40 with steps of 5 

#Multiplying it with 10^-3 to convert into msec

axes1.set_xticklabels(np.arange(0,41,5), fontsize=14);



axes1.set_yticks([-3,-2,-1,0,1,2,3])

axes1.set_yticklabels([-3,-2,-1,0,1,2,3], fontsize=14);



axes1.set_xlabel("time(ms)", fontsize=18);

axes1.set_ylabel("Amplitude", fontsize=18);

axes1.set_title("Signal Vs time", fontsize=18)
#To sample this signal

Fsampling=1000

#Applying Nyquist Crieteria 

ts=1/Fsampling #1ms



txs=np.arange(0,(time+ts/2), ts)

#creating an array that notes sampling pt of time from 0 to 40ms 

r=np.round(len(t)/len(txs))#indexing the sample (0, r, 2r, 3r...)

#Total number of Samples 



xts=np.arange(0, len(t), r). astype('int')

#Contains the index of r



xs=x[xts] # contains value of x at r index

ys=yc[xts] #contains value of y at r index
#Visualising the Signal with added random noise

#To create an empty figure

fig1= plt.figure(figsize=(10,4)) #(length, height)

axes1=fig1.add_axes([0.1,0.1,0.8,0.8]) 

#Plotting

axes1.plot(x,yc, color='red', linewidth=3, linestyle='-')

axes1.plot(xs, ys, color='blue', linestyle='', marker='o')

axes1.bar(xs, ys, bottom=0, width=0.05, color='black')

#creating the verticle lines from sampling dot to Dc Line

axes1.axhline(0, color='black', linestyle='-', linewidth=3)

#creating the zero DC Line



axes1.set_ylim([-3,-3])

axes1.set_xlim([0,np.max(x)])



axes1.set_xticks((2*math.pi*freq)*(np.arange(0,41,5)*1e-3));

#Creating an array from 0 to 40 with steps of 5 

#Multiplying it with 10^-3 to convert into msec

axes1.set_xticklabels(np.arange(0,41,5), fontsize=14);



axes1.set_yticks([-3,-2,-1,0,1,2,3])

axes1.set_yticklabels([-3,-2,-1,0,1,2,3], fontsize=14);



axes1.set_xlabel("time(ms)", fontsize=18);

axes1.set_ylabel("Amplitude", fontsize=18);

axes1.set_title("Signal Vs time", fontsize=18)
#Check the samples tacken

ys
#Rechecking with a different Sampling Frequency

#To sample this signal

Fsampling=500

#Applying Nyquist Crieteria 

ts=1/Fsampling #1ms



txs=np.arange(0,(time+ts/2), ts)

#creating an array that notes sampling pt of time from 0 to 40ms 

r=np.round(len(t)/len(txs))#indexing the sample (0, r, 2r, 3r...)

#Total number of Samples 



xts=np.arange(0, len(t), r). astype('int')

#Contains the index of r



xs=x[xts] # contains value of x at r index

ys=yc[xts] #contains value of y at r index



#Visualising the Signal with added random noise

#To create an empty figure

fig1= plt.figure(figsize=(10,4)) #(length, height)

axes1=fig1.add_axes([0.1,0.1,0.8,0.8]) 

#Plotting

axes1.plot(x,yc, color='red', linewidth=3, linestyle='-')

axes1.plot(xs, ys, color='blue', linestyle='', marker='o')

axes1.bar(xs, ys, bottom=0, width=0.05, color='black')

#creating the verticle lines from sampling dot to Dc Line

axes1.axhline(0, color='black', linestyle='-', linewidth=3)

#creating the zero DC Line



axes1.set_ylim([-3,-3])

axes1.set_xlim([0,np.max(x)])



axes1.set_xticks((2*math.pi*freq)*(np.arange(0,41,5)*1e-3));

#Creating an array from 0 to 40 with steps of 5 

#Multiplying it with 10^-3 to convert into msec

axes1.set_xticklabels(np.arange(0,41,5), fontsize=14);



axes1.set_yticks([-3,-2,-1,0,1,2,3])

axes1.set_yticklabels([-3,-2,-1,0,1,2,3], fontsize=14);



axes1.set_xlabel("time(ms)", fontsize=18);

axes1.set_ylabel("Amplitude", fontsize=18);

axes1.set_title("Signal Vs time", fontsize=18)