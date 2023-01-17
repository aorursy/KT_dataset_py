import numpy as np

import matplotlib.pylab as plt

from scipy.stats import uniform, norm, laplace, beta, bernoulli, expon

#import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf

import warnings 

import pystan

import arviz

warnings.filterwarnings('ignore')
ndat = 1000

p = 0.7

x = bernoulli.rvs(p=p, size=ndat)



print(f'Number of ones: {x.sum()}.')

print(f'Number of zeros: {ndat - x.sum()}')

print(f'Estimated probability: {x.sum()/ndat:.3f}')
model_code = """

data {

    int<lower=0> N;                    // number of data items

    int<lower=0,upper=1> x[N];         // data (observations)

}



parameters {

    real<lower=0,upper=1> p;           // estimated variable

}



model {

    p ~ beta(1, 1);  // prior for the mean

    

    for(n in 1:N) {

        x[n] ~ bernoulli(p); // data model

    }

}

"""



sm = pystan.StanModel(model_code=model_code)
data = {

    'x': x,

    'N': ndat

}
fit = sm.sampling(data=data, iter=100000, chains=1, verbose=True)
print(fit)
inferencedata = arviz.from_pystan(posterior=fit)
arviz.plot_trace(inferencedata)
trace = fit.extract(pars='p')
a = b = 1

a1 = a + x.sum()

b1 = b + (ndat - x.sum())



xticks = np.linspace(0.68, 0.8, 1000)

from scipy.stats import gaussian_kde

kde_res = gaussian_kde(trace['p'])

plt.hist(trace['p'], density=True, color='lightblue')

plt.plot(xticks, kde_res(xticks))



plt.plot(xticks, beta.pdf(xticks, a1, b1))
y = np.loadtxt('../input/ucuss/housefly.txt')

plt.hist(y, density=True)

plt.title(f'Housefly wing lengths, N={y.size}, mean={y.mean():.2f}, std={y.std():.2f}')

plt.show()
model_code = """

data {

    int<lower=0> N;      // number of data items

    vector[N] y;         // data (observations)

}



parameters {

    real mu;             // estimated mean

    real<lower=0> std;   // estimated standard dev.

}



model {

    mu ~ normal(50, 100); // prior for the mean

    std ~ gamma(2, 0.1);  // prior for st. dev.

    y ~ normal(mu, std);  // data model

}

"""



sm_housefly = pystan.StanModel(model_code=model_code)
data = {

    'y': y,

    'N': y.size

}



fit_housefly = sm_housefly.sampling(data=data, iter=100000, chains=1, verbose=True)

print(fit_housefly)
inferencedata_housefly = arviz.from_pystan(posterior=fit_housefly)

arviz.plot_trace(inferencedata_housefly)
data = np.load('../input/ucuss/time-to-event_1.npz')

y = data['y']

plt.hist(y, density=True)
model_code = """

data {

    int<lower=0> N;          // number of data items

    vector[N] y;             // data (observations)

}



parameters {

    real<lower=0> lambda;    // estimated rate parameter

}



model {

    lambda ~ gamma(0.001, 0.001);  // prior for lambda

    y ~ exponential(lambda);       // data model

}



generated quantities {             // here we calculate posterior predictive

    real yhat[N];                  // this is the predicted observation

    

    for (n in 1:N) {

         yhat[n] = exponential_rng(lambda);    // we sample it from the rng with lambda est.

    }

}

"""



sm_tte = pystan.StanModel(model_code=model_code)
data = {

    'y': y,

    'N': y.size

}



fit_tte = sm_tte.sampling(data=data, iter=100000, chains=1, verbose=True)
print(fit_tte.stansummary(pars='lambda'))
inferencedata_tte = arviz.from_pystan(posterior=fit_tte, posterior_predictive='yhat', observed_data='y')
arviz.plot_trace(inferencedata_tte)
arviz.plot_ppc(inferencedata_tte, data_pairs = {'y': 'yhat'}, alpha=0.9)