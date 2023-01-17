# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import scipy.stats as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

x_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

LabelEncoder_x = LabelEncoder()

label = LabelEncoder_x.fit_transform(y_train.surface)

x_test = pd.read_csv('../input/X_test.csv')
from numpy.lib.stride_tricks import as_strided

def stft(x, fftsize=64, overlap_pct=.5):   

    hop = int(fftsize * (1 - overlap_pct))

    w = scipy.hanning(fftsize + 1)[:-1]    

    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])

    return raw[:, :(fftsize // 2)]

def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):

    win_size = l_size + r_size + c_size

    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)

    strides = x.strides + (x.strides[-1],)

    xs = as_strided(x, shape=shape, strides=strides)

    def is_peak(x):

        centered = (np.argmax(x) == l_size + int(c_size/2))

        l = x[:l_size]

        c = x[l_size:l_size + c_size]

        r = x[-r_size:]

        passes = np.max(c) > np.max([f(l), f(r)])

        if centered and passes:

            return np.max(c)

        else:

            return -1

    r = np.apply_along_axis(is_peak, 1, xs)

    top = np.argsort(r, None)[::-1]

    heights = r[top[:n_peaks]]



    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)

    return heights, top[:n_peaks]
import scipy

columns = ['orientation_X', 'orientation_Y','orientation_Z','orientation_W', 'angular_velocity_X','angular_velocity_Y','angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

all_obs = []

for ide in x_train.series_id.unique():

    d = np.abs(stft(x_train[x_train['series_id']==ide][columns[0]]))

    for column in columns[1:]:

        pic = np.abs(stft(x_train[x_train['series_id']==ide][column]))

        d = np.concatenate((d, pic), axis=0)

    n_dim = 6

    obs = np.zeros((n_dim, d.shape[0]))

    for r in range(d.shape[0]):

        _, t = peakfind(d[r, :], n_peaks=n_dim)

        obs[:, r] = t.copy()



    all_obs.append(obs)

    

all_obs = np.atleast_3d(all_obs)
x_test_obs = []

for ide in x_test.series_id.unique():

    d = np.abs(stft(x_test[x_test['series_id']==ide][columns[0]]))

    for column in columns[1:]:

        pic = np.abs(stft(x_test[x_test['series_id']==ide][column]))

        d = np.concatenate((d, pic), axis=0)

    n_dim = 6

    obs = np.zeros((n_dim, d.shape[0]))

    for r in range(d.shape[0]):

        _, t = peakfind(d[r, :], n_peaks=n_dim)

        obs[:, r] = t.copy()



    x_test_obs.append(obs)

    

x_test_obs = np.atleast_3d(x_test_obs)
class gmmhmm:

    # This class converted with modifications from:

    # https://code.google.com/p/hmm-speech-recognition/source/browse/Word.m

    def __init__(self, n_states):

        self.n_states = n_states

        self.random_state = np.random.RandomState(0)

        

        # Normalize random initial state

        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))

        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))

        

        self.mu = None

        self.covs = None

        self.n_dims = None

           

    def _forward(self, B):

        log_likelihood = 0.

        T = B.shape[1] # Number of observation

        alpha = np.zeros(B.shape)

        for t in range(T):

            if t == 0:

                alpha[:, t] = B[:, t] * self.prior.ravel()

            else:

                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])

         

            alpha_sum = np.sum(alpha[:, t])

            alpha[:, t] /= alpha_sum

            log_likelihood = log_likelihood + np.log(alpha_sum)

        return log_likelihood, alpha

    

    def _backward(self, B):

        T = B.shape[1]

        beta = np.zeros(B.shape);

           

        beta[:, -1] = np.ones(B.shape[0])

            

        for t in range(T - 1)[::-1]:

            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))

            beta[:, t] /= np.sum(beta[:, t])

        return beta

    

    def _state_likelihood(self, obs):

        obs = np.atleast_2d(obs)

        B = np.zeros((self.n_states, obs.shape[1]))

        for s in range(self.n_states):

            np.random.seed(self.random_state.randint(1))

            B[s, :] = st.multivariate_normal.pdf(

                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T) 

            # This is a statstical density func rather than a pdf

        return B

    

    def _normalize(self, x):

        return (x + (x == 0)) / np.sum(x)

    

    def _stochasticize(self, x):

        return (x + (x == 0)) / np.sum(x, axis=1)

    

    def _em_init(self, obs):

        # Using this _em_init function allows for less required constructor args

        if self.n_dims is None:

            self.n_dims = obs.shape[0]

        if self.mu is None:

            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=True)

            self.mu = obs[:, subset]

        if self.covs is None:

            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))

            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]

        return self

    

    def _em_step(self, obs): 

        obs = np.atleast_2d(obs)

        B = self._state_likelihood(obs)

        T = obs.shape[1]

        

        log_likelihood, alpha = self._forward(B)

        beta = self._backward(B)

        

        xi_sum = np.zeros((self.n_states, self.n_states))

        gamma = np.zeros((self.n_states, T))

        

        for t in range(T - 1):

            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)

            xi_sum += self._normalize(partial_sum)

            partial_g = alpha[:, t] * beta[:, t]

            gamma[:, t] = self._normalize(partial_g)

              

        partial_g = alpha[:, -1] * beta[:, -1]

        gamma[:, -1] = self._normalize(partial_g)

        

        expected_prior = gamma[:, 0]

        expected_A = self._stochasticize(xi_sum)

        

        expected_mu = np.zeros((self.n_dims, self.n_states))

        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))

        

        gamma_state_sum = np.sum(gamma, axis=1)

        # Set zeros to 1 before dividing

        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)

        

        for s in range(self.n_states):

            gamma_obs = obs * gamma[s, :]

            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]

            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)

            #Symmetrize

            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)

        

        # Ensure positive semidefinite by adding diagonal loading

        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]

        

        self.prior = expected_prior

        self.mu = expected_mu

        self.covs = expected_covs

        self.A = expected_A

        return log_likelihood

    

    def fit(self, obs, n_iter=15):

        count = obs.shape[0]

        for n in range(count):

            for i in range(n_iter):

                self._em_init(obs[n, :, :])

                log_likelihood = self._em_step(obs[n, :, :])

        return self

    

    def transform(self, obs):

        # Support for 2D and 3D arrays

        # 2D: (n_features, n_dims)

        # 3D: (n_examples, n_features, n_dims)

        if len(obs.shape) == 2:

            B = self._state_likelihood(obs)

            log_likelihood, _ = self._forward(B)

            return log_likelihood

        elif len(obs.shape) == 3:

            count = obs.shape[0]

            out = np.zeros((count,))

            for n in range(count):

                B = self._state_likelihood(obs[n, :, :])

                log_likelihood, _ = self._forward(B)

                out[n] = log_likelihood

            return out
ys = set(label)

ms = [gmmhmm(9) for y in ys]

_ = [m.fit(all_obs) for m in ms]

ps = [m.transform(x_test_obs) for m in ms]

res = np.vstack(ps)

predicted_labels = np.argmax(res, axis=0)
predicted_labels = LabelEncoder_x.inverse_transform(predicted_labels)
import time

from datetime import datetime

ver = 'GMMHMM'

filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))

pd.DataFrame({

    'series_id': x_test.series_id.unique(),

    'surface': predicted_labels

}).to_csv(filename+'1'+'.csv', index=False)