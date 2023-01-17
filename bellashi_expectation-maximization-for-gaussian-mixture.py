# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import uniform

from scipy.special import logsumexp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class Gaussian:

    def __init__(self, mu, sigma):

        # mean and standard deviation

        self.mu = mu

        self.sigma = sigma



    def pdf(self, datum):

        "probability of a data point given the current parameters"

        u = (datum - self.mu) / abs(self.sigma)

        y = (1 / (np.sqrt(2 * np.pi) * abs(self.sigma))) * np.exp(-u * u / 2)

        return y



    def log_pdf(self, datum):

        "log probability of a data point given the current parameters"

        u = (datum - self.mu) / abs(self.sigma)

        y = np.log((1 / (np.sqrt(2 * np.pi) * abs(self.sigma)))) + (-u * u / 2)

        return y

class GaussianMixture:

    """

        using numpy package for computation

    """

    def __init__(self, n_components):

        # list of gaussian components

        self.g = None



        # weights of the gaussian components

        self.mix = None



        # the number of mixture components.

        self.n_components = n_components



        # the convergence threshold.

        # em iterations will stop when the lower bound average gain is below this threshold.

        self.tol = 0.001



        # number of step used by the best fit of EM to reach the convergence.

        self.n_iter_ = None



    def init_model(self, X):

        mu_min = min(X)

        mu_max = max(X)

        sigma_min = 1

        sigma_max = 1



        g = []

        mix = []

        for i in range(self.n_components):

            g.append(Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max)))

            mix.append(1 / self.n_components)



        self.g = g

        self.mix = mix



        return self



    def e_step(self, X):

        """e step.



        Parameters

        ----------

        X : array-like, shape (n_samples,)



        Returns

        -------

        log_prob_norm : float

            Mean of the logarithms of the probabilities of each sample in X



        log_responsibility : array, shape (n_samples, n_components)

            Logarithm of the posterior probabilities (or responsibilities) of

            the point of each sample in X.

        """

        assert X is not None and len(X) > 0, 'X is none or empty'

        assert self.g is not None and len(self.g) > 0, 'g is none or empty'

        assert self.mix is not None and len(self.mix) > 0, 'mix is none or empty'

        assert len(self.g) == len(self.mix), 'length of g and mix is not equal'



        log_prob_norm, log_resp = self.estimate_log_prob_resp(X)

        return np.mean(log_prob_norm), log_resp



    def estimate_log_prob_resp(self, X):

        """Estimate log probabilities and responsibilities for each sample.



        Refer to function "_estimate_log_prob_resp()" in sklearn\mixture\base.py



        Compute the log probabilities, weighted log probabilities per

        component and responsibilities for each sample in X with respect to

        the current state of the model.



        Parameters

        ----------

        X : array-like, shape (n_samples,)



        Returns

        -------

        log_prob_norm : array, shape (n_samples,)

            log p(X)



        log_responsibilities : array, shape (n_samples, n_components)

            logarithm of the responsibilities

        """



        # weighted_log_prob : array, shape (n_samples, n_components)

        weighted_log_prob = self.estimate_weighted_log_prob(X)



        # log_prob_norm: array, shape (n_samples,)

        #       i.e., log p(X)

        log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        assert len(log_prob_norm) == len(X), 'length of log_prob_norm error'



        with np.errstate(under='ignore'):

            # ignore underflow

            # log_resp : array, shape (n_samples, n_components)

            #       logarithm of the responsibilities

            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]



        return log_prob_norm, log_resp



    def estimate_weighted_log_prob(self, X):

        # refer to function "_estimate_weighted_log_prob()" in sklearn\mixture\base.py

        # estimate the weighted log-probabilities, log P(X | Z) + log weights.

        # weighted_log_prob : array, shape (n_samples, n_components)

        weighted_log_prob = []

        for i in range(len(self.g)):

            a = [self.g[i].log_pdf(x) + np.log(self.mix[i]) for x in X]

            # here we assume X is 1D array

            assert len(a) == len(X), 'length of array a error'

            weighted_log_prob.append(a)



        weighted_log_prob = np.array(weighted_log_prob)

        return weighted_log_prob.T



    def m_step(self, X, log_resp):

        """M step.



        Parameters

        ----------

        X : array-like, shape (n_samples,)



        log_resp : array-like, shape (n_samples, n_components)

            Logarithm of the posterior probabilities (or responsibilities) of

            the point of each sample in X.

        """



        # resp : array-like, shape (n_samples, n_components)

        #   The responsibilities for each data sample in X.

        resp = np.exp(log_resp)



        # nk : array-like, shape (n_components,)

        #   The numbers of data samples in the current components.

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps



        resp = resp.T



        # compute new means

        for i in range(len(self.g)):

            self.g[i].mu = np.dot(resp[i], np.array(X)) / nk[i]



        # compute new sigmas

        for i in range(len(self.g)):

            self.g[i].sigma = np.sqrt(np.dot(resp[i], (np.array(X) - self.g[i].mu) ** 2) / nk[i])



        # compute new mix

        for i in range(len(self.g)):

            self.mix[i] = nk[i] / len(X)



    def pdf(self, x):

        v = 0

        for i in range(len(self.g)):

            v += self.g[i].pdf(x) * self.mix[i]

        return v



    def fit(self, X, max_iter):

        "perform n iterations, then compute log-likelihood"

        lower_bound = None



        for i in range(max_iter):

            self.n_iter_ = i

            print("iter: " + str(i))

            prev_lower_bound = lower_bound



            # if verbose:

            #     print('iteration: ' + str(i))



            log_prob_norm, log_resp = self.e_step(X)

            self.m_step(X, log_resp)



            lower_bound = log_prob_norm



            if prev_lower_bound is not None:

                change = lower_bound - prev_lower_bound

                if abs(change) < self.tol:

                    break

X1 = np.random.normal(2, 2, 10000)

X2 = np.random.normal(10, 2, 5000)

X=np.append(X1,X2)



n_components = 3

max_iter = 100



model = GaussianMixture(n_components)

model.init_model(X)

model.fit(X, max_iter)
import matplotlib.pyplot as plt

import seaborn as sns

    

def plot_data(data, model):

    # plot

    min_graph = min(data)

    max_graph = max(data)

    x = np.linspace(min_graph, max_graph, 2000)  # to plot the data



    sns.distplot(data, bins=20, kde=False, norm_hist=True)



    g_both = [model.pdf(e) for e in x]

    plt.plot(x, g_both, label='gaussian mixture')
plot_data(X, model)