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



fs = 10000

f0 = 100



K = np.int(fs/f0)

a1 = -0.99 

x['x0'] = 0.

x['x0'] = x['signal'] + a1 * x['signal'].shift(K)

x.loc[0:K-1,'x0'] = x.loc[0:K-1,'signal']
# Energy of a signal is i^2 * time

dt = 0.0001



x.loc[:,'energy'] = x['x0']**2 * dt



# energy of our signal = energy of measurement + injection energy

# measurement energy

x['x1'] = x['energy'] - x['energy'].rolling(window=7500,min_periods=5).mean()

x.loc[0:4,'x1'] = x.loc[0:4,'energy']



# The energy_floor over 7500 periods (0.75s) is

# is it injection energy ??

x['x2']  = - x['x1'].rolling(window=7500, min_periods=5).min()

x.loc[0:4,'x2'] = 0.    #   - x.loc[0:4,'x1']



# injection current

x['x2'] = np.sqrt(x['x2']) / dt



# x2 will denote the mode of operation. Mode changes very infrequently



examples = ['signal','y','x0','x1','x2']



fig, ax = plt.subplots(nrows=len(examples), ncols=1, figsize=(25, 3.5*len(examples)))

fig.subplots_adjust(hspace = .5)

ax = ax.ravel()

colors = plt.rcParams["axes.prop_cycle"]()



for i in range(len(examples)):

    

    c = next(colors)["color"]

    ax[i].grid()

    if examples[i] in ['x0','x2','signal']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= 2)

        ax[i].set_ylabel('current (pA)', fontsize=14)

        

    if examples[i] in ['y']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= .1)

        ax[i].set_ylabel('Open Channels', fontsize=14)

    if examples[i] in ['x1']:

        ax[i].plot(x['time'], x[examples[i]],color=c, linewidth= .1)

        ax[i].set_ylabel('Energy 10^-24 W-s', fontsize=14)                     

    ax[i].plot(x['time'], x[examples[i]],color=c, linewidth=.5)

    ax[i].set_title(examples[i], fontsize=24)

    ax[i].set_xlabel('Time (seconds)', fontsize=14)

    #ax[i].set_ylabel('current (pA)', fontsize=24)

    #ax[i].set_ylim(0,5)
from collections import namedtuple

gaussian = namedtuple('Gaussian', ['mean', 'var'])

gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])
def update(prior, measurement):

    x, P = prior        # mean and variance of prior of x (system)

    z, R = measurement  # mean and variance of measurement (open_channels) with ion probe

    

    J = 1 - f1_score(z,y)        # residual - This is error we want to minumize

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

P = np.eye(3) * 2

R = [1,1]

dim_x = 3

for i in range(1000):

    

    for j,k in enumerate(['x0','x1','x2']):

        measurement = gaussian(x.loc[i,'y'],R)

        

        x[k], P[:,j] = update(P[:,j],x.loc[i,'y'])

        x[k], P[:,j] = predict(P[:,j],x[k])

    x.loc[i,'y_pred'] = x.mean

    
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
# Reference https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

def macro_double_soft_f1(y, y_hat):

    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).

    Use probability values instead of binary predictions.

    This version uses the computation of soft-F1 for both positive and negative class for each label.

    

    Args:

        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)

        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        

    Returns:

        cost (scalar Tensor): value of the cost function for the batch

    """

    y = tf.cast(y, tf.float32)

    y_hat = tf.cast(y_hat, tf.float32)

    tp = tf.reduce_sum(y_hat * y, axis=0)

    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)

    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)

    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)

    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1

    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0

    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0

    macro_cost = tf.reduce_mean(cost) # average on all labels

    return macro_cost
train_df['work'] = train_df['signal']**2 - (train_df['signal']**2).mean()

pairs = pd.DataFrame(window(train_df.loc[:,'work']), columns=['state1', 'state2'])

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