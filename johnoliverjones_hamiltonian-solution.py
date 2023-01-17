import numpy as np

from numpy.fft import *

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

import scipy as sp

import scipy.fftpack

from scipy import signal

from pykalman import KalmanFilter

from sklearn import tree

import tensorflow as tf

import tensorflow_hub as hub



import gc



from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split





DATA_PATH = "../input/liverpool-ion-switching"



x = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

#test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

#submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
x = x.rename(columns = {'open_channels':'y'})

dt = 0.0001

fs = 10000

f0 = 100



K = np.int(fs/f0)

a1 = -0.99 

x['x0'] = 0.

x['x0'] = x['signal'] + a1 * x['signal'].shift(K)

x.loc[0:K-1,'x0'] = x.loc[0:K-1,'signal']



x.loc[:,'signal_energy'] = x['x0']**2 * dt

x.loc[:,'original_energy'] = x['signal']**2 * dt



filter_gain =  np.sqrt( x['signal_energy'].sum()/x['original_energy'].sum()) 

print(f'filter gain is {filter_gain:.3f}')

print(f'filter loss is {1./filter_gain:.3f}')

r = 350. # units are Ohms

r1 = 3. # current source resistance (est)

a1 = 1/filter_gain  # loss in estimate of injection current)



x.loc[:,'signal_energy'] = a1 * (r+r1)/(r*r1) *  x['x0']**2 * dt
# Energy of a signal is r i^2 * time





dt = 0.0001

fs = 1. / dt





# energy of our signal = energy of message + injection energy

# the injection current is DC, so we must use original siggnal before filtering 

# to find the DC offset (aka injection current). Assume it is a square wave as described in literature.



x['injected_current'] = a1 * x['x0'].rolling(window=7500,min_periods=5).min()

x.loc[0:4,'injected_current'] = 0

x['injected_energy'] = r1 * x['injected_current']**2 * dt/4 # I think injected current is square wave

x.loc[0:4,'injected_energy']  = 0.



# energy of message = energy of signal - injection energy

# message energy



x= x.fillna(0)

x['message_current'] = x['x0'] - x['injected_current']

x['message_energy'] = r * x['message_current']**2 * dt



# calculate the mean of open channels as a moving averge

x['y_mean'] = x['y'].rolling(window=100,min_periods=5).mean()

x.loc[0:4,'y_mean'] = 0


examples = ['signal','y','y_mean','signal_energy','injected_energy','message_energy','x0','injected_current','message_current']



fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 3.5*len(examples)))

fig.subplots_adjust(hspace = .5)

ax = ax.ravel()

colors = plt.rcParams["axes.prop_cycle"]()



for i in range(len(examples)):

    

    c = next(colors)["color"]

    ax[i].grid()

    if examples[i] in ['x0','signal','message_current','injected_current']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)

        ax[i].set_ylabel('current (pA)', fontsize=14)

        

    if examples[i] in ['y','y_mean']:

        ax[i].scatter(x['time'][::10], x[examples[i]][::10],marker ='.', color=c)

        ax[i].set_ylabel('Open Channels', fontsize=14)

    if examples[i] in ['signal_energy','energy_inj','message_energy','injected_energy']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 1)

        ax[i].set_ylabel('Energy 10^-24 Joules', fontsize=14)                     

    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)

    ax[i].set_title(examples[i], fontsize=24)

    ax[i].set_xlabel('Time (seconds)', fontsize=14)

    #ax[i].set_ylabel('current (pA)', fontsize=24)

    #ax[i].set_ylim(0,5)
plt.close()

# calculate the average energy per change in open channels



x['y_var'] = x['y'] - x['y'].shift(1)

x.loc[0,'y_var'] = 0



# when a change in message energy occurs



x['dE'] = (x['message_energy'] - x['message_energy'].shift(1))

x.loc[0,'dE'] = 0

de = x.loc[x.dE != 0.,['dE']].values

dy = x.loc[x.dE != 0.,['y_var']].values



# measure k = (change in y) / (energy change)

k = np.mean(dy/de)

k2 = np.mean(np.abs(dy/de))

print(f'{k:.3e} ion transitions per 10^-24 J')

print(f'{k2:.3e} ion transitions per J --- k2')



# The message current is the flow of ions in the probe. It should be a direct measure of open channels

x['y_est'] = r * x['message_current']**2 * k * dt * 5

# the constant 6 makes better solution, but why 6?

x['y_est'] = x['y_est'].clip(0,10)

# calculate the mean of open channels as a moving averge

x['y_mean_est'] = x['y_est'].rolling(window=100,min_periods=5).mean()

x.loc[0:4,'y_mean'] = 0



x['y_var_est'] = x['y_est']- x['y_est'].shift(1)

x.loc[0,'y_var'] = 0





examples = ['signal','y','y_est', 'y_var','y_var_est','y_mean','y_mean_est']



fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 3.5*len(examples)))

fig.subplots_adjust(hspace = .5)

ax = ax.ravel()

colors = plt.rcParams["axes.prop_cycle"]()



for i in range(len(examples)):

    

    c = next(colors)["color"]

    ax[i].grid()

    if examples[i] in ['x0','signal','message_current']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)

        ax[i].set_ylabel('current (pA)', fontsize=14)

        

    if examples[i] in ['y','y_var','y_mean','y_var']:

        ax[i].scatter(x['time'][::10], x[examples[i]][::10],marker ='.', color=c, linewidth=0)

        ax[i].set_ylabel('Open Channels', fontsize=14)

    if examples[i] in ['y_mean_est','y_pred']:

        ax[i].scatter(x['time'][::10], (x[examples[i]][::10]).round(0),marker ='.', color=c, linewidth=0)

        ax[i].set_ylabel('Open Channels', fontsize=14)

        #ax[i].set_ylim(0,10)

    if examples[i] in ['y_est']:

        ax[i].scatter(x['time'][::10], (x[examples[i]][::10]).round(0),marker ='.', color=c, linewidth=0)

        ax[i].set_ylabel('Open Channels', fontsize=14)

        #ax[i].set_ylim(0,10)

    if examples[i] in ['y_var_est']:

        ax[i].scatter(x['time'][::10], (x[examples[i]][::10]).round(0),marker ='.', color=c, linewidth=0)

        ax[i].set_ylabel('Open Channels', fontsize=14)

    if examples[i] in ['signal_energy','injected_energy','message_energy']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 1)

        ax[i].set_ylabel('Energy 10^-24 Joules', fontsize=14)                     

    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)

    ax[i].set_title(examples[i], fontsize=24)

    ax[i].set_xlabel('Time (seconds)', fontsize=14)

    #ax[i].set_ylabel('current (pA)', fontsize=24)

    #ax[i].set_ylim(0,5)
plt.close()

# Calculate the f1_score for the estimate

# first descretize the estimate by rounding to ones digit.



x['y_est'] = x['y_est'].round(0)



f1 = f1_score(x.y, x.y_est, average='macro')

print(f'f1_score is {f1:.3f}')
# Reference 3

from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])

gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])



def update(prior, measurement):

    x, P = prior        # mean and variance of prior of x (system)

    z, R = measurement  # mean and variance of measurement (open_channels) with ion probe

    

    J = z-x          #1 - f1_score(z,x)        # residual - This is error we want to minumize

    K = P / (P + R)              # Kalman gain



    x = x + K*J      # posterior

    P = (1 - K) * P  # posterior variance

    return gaussian(x, P)



def predict(posterior, movement):

    x, P = posterior # mean and variance of posterior

    dx, Q = movement # mean and variance of movement

    x = x + dx

    P = P + Q

    return gaussian(x, P)

R = 1

Q = 2.

P = gaussian(0,R)



for i in range(1000):

    measurement = gaussian(x.loc[i,'y_est'],R)        

    est, P = update(P,measurement)

    

    movement = gaussian(x.loc[i,'y'],Q)

    est, P = predict(P,movement)

    x.loc[i,'y_pred_k'] = est

    
from itertools import islice



def window(seq, n=2):

    "Sliding window width n from seq.  From old itertools recipes."""

    it = iter(seq)

    result = tuple(islice(it, n))

    if len(result) == n:

        yield result

    for elem in it:

        result = result[1:] + (elem,)

        yield result

        

pairs = pd.DataFrame(window(x.loc[:,'y']), columns=['state1', 'state2'])

counts = pairs.groupby('state1')['state2'].value_counts()

alpha = 1 # Laplacian smoothing is when alpha=1

counts = counts + 1

#counts = counts.fillna(0)

P = ((counts + alpha )/(counts.sum()+alpha)).unstack()

P


pairs = pd.DataFrame(window(x.loc[:,'signal_energy']), columns=['state1', 'state2'])

means = pairs.groupby('state1')['state2'].mean()

alpha = 1 # Laplacian smoothing is when alpha=1

means = means.unstack()

means
print('Occurence Table of State Transitions')

ot = counts.unstack().fillna(0)

ot
P = (ot)/(ot.sum())

Cal = - P * np.log(P)

Cal
Caliber = Cal.sum().sum()

Caliber
# reference https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data

def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):

    fig, axes = plt.subplots(numplots_y, numplots_x)

    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig, axes



fig, axes = create_axes_grid(1,1,10,5)

axes.set_title('Markov Transition Matrix P for all of train')

sns.heatmap(

    P,

    annot=True, fmt='.3f', cmap='Blues', cbar=False,

    ax=axes, vmin=0, vmax=0.5, linewidths=2);
eig_values, eig_vectors = np.linalg.eig(np.transpose(P))

print("Eigenvalues :", eig_values)
# reference: http://kitchingroup.cheme.cmu.edu/blog/2013/02/03/Using-Lagrange-multipliers-in-optimization/

def func(X):

    x = X[0]

    y = X[1]

    L = X[2] 

    return x + y + L * (x**2 + k * y)



def dfunc(X):

    dL = np.zeros(len(X))

    d = 1e-4 

    for i in range(len(X)):

        dX = np.zeros(len(X))

        dX[i] = d

        dL[i] = (func(X+dX)-func(X-dX))/(2*d);

    return dL
from scipy.optimize import fsolve



# this is the max

X1 = fsolve(dfunc, [1, 1, 0])

print(X1, func(X1))



# this is the min

X2 = fsolve(dfunc, [-1, -1, 0])

print(X2, func(X2))
del x