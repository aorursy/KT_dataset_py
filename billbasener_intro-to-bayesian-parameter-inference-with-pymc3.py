# This is a simple example of using pymc3 for Bayesian inference of the parameter distribution.
# It was adopted from Simon Ouellette's Intro to PyMC3:
# https://github.com/SimonOuellette35/Introduction_to_PyMC3

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

obs_y = np.random.normal(0.5, 0.35, 5000)
with pm.Model() as exercise1:

    # prior probabilitiy distributions
    stdev = pm.HalfNormal('stdev', sd=0.05)
    mu = pm.Normal('mu', mu=0.0, sd=0.05)

    # The model  - y is a normal distribution with mean mu and standard deviation  stdev.
    y = pm.Normal('y', mu=mu, sd=stdev, observed=obs_y)

    # Iterate MCMC
    trace = pm.sample(1000)

pm.traceplot(trace, ['mu', 'stdev'])
plt.show()