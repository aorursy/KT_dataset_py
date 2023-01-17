import pymc3 as pm

import numpy as np

import pandas as pd

import theano.tensor as tt

from scipy.special import expit

invlogit = lambda x: 1/(1 + tt.exp(-x))
df = pd.read_csv('../input/train.csv')

y = np.asarray(df.target)

X = np.array(df.ix[:, 2:302])

df2 = pd.read_csv('../input/test.csv')

df2.head()

X2 = np.array(df2.ix[:, 1:301])
def get_model(y, X):

    model = pm.Model()

    with model:

        xi = pm.Bernoulli('xi', .05, shape=X.shape[1]) #inclusion probability for each variable

        alpha = pm.Normal('alpha', mu = 0, sd = 5) # Intercept

        beta = pm.Normal('beta', mu = 0, sd = .75 , shape=X.shape[1]) #Prior for the non-zero coefficients

        p = pm.math.dot(X, xi * beta) #Deterministic function to map the stochastics to the output

        y_obs = pm.Bernoulli('y_obs', invlogit(p + alpha),  observed=y)  #Data likelihood

    return model
model1 = get_model(y, X)
with model1:

    trace = pm.sample(2000, random_seed = 4816, cores = 1, progressbar = True, chains = 1)
results = pd.DataFrame({'var': np.arange(300), 

                        'inclusion_probability':np.apply_along_axis(np.mean, 0, trace['xi']),

                       'beta':np.apply_along_axis(np.mean, 0, trace['beta']),

                       'beta_given_inclusion': np.apply_along_axis(np.sum, 0, trace['xi']*trace['beta'])

                            /np.apply_along_axis(np.sum, 0, trace['xi'])

                       })
results.sort_values('inclusion_probability', ascending = False).head(20)
#Scoring test.  Score new data from a single posterior sample

test_beta = trace['beta'][0]

test_inc = trace['xi'][0]

test_score = expit(trace['alpha'][0] + np.dot(X2, test_inc * test_beta))  

test_score
estimate = trace['beta'] * trace['xi'] 

y_hat = np.apply_along_axis(np.mean, 1, expit(trace['alpha'] + np.dot(X2, np.transpose(estimate) )) )
#Sanity checks

np.mean(y_hat), np.sum(results.inclusion_probability/300)
submission  = pd.DataFrame({'id':df2.id, 'target':y_hat})

submission.to_csv('submission.csv', index = False)