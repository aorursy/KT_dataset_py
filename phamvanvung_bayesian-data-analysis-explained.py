import numpy as np

import pandas as pd
n_sample = 16

n_registered = 6


n_draws = 10000

# samples randomly for the prior

prior = pd.Series(np.random.uniform(0.0, 1.0, n_draws))

# draw the distributions of the prior

prior.hist()
# This is either 0, 1 (binary) rate so it should follow Bernoulli distribution

# In this case parameter is the one of the randomly generated success rate based on prior in the previous step

def generative_model(parameters):

    return np.random.binomial(n_sample, parameters)
# Test

print(generative_model(0.1))

print(generative_model(0.9))
sim_data = list()

for p in prior:

    sim_data.append(generative_model(p))
posterior = prior[[x == n_registered for x in sim_data]]
posterior
# show the distribution of the posterior

posterior.hist()
print(f"Number of draws left: {len(posterior)}")
print(f"Posterior median: {'%.3f'%posterior.median()}")
print(f"Posterior interval: {'%.3f'%posterior.quantile(0.25)}, {'%.3f'%posterior.quantile(0.75)}")
# how is the result comparing to 20%, meaning the percentages of posteriors that we can have the registration rate as 20%

sum(posterior >0.2)/len(posterior)
registrations = pd.Series([np.random.binomial(n=100, p=p) for p in posterior])
registrations.hist()
print('Registration 95%% quantile interval %d-%d'%tuple(registrations.quantile([0.025, 0.975]).values))