import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import pymc3 as pm

print(pm.__version__)
d = np.array([3, 2, 1])

alphas = np.array([1.5, 1, 0.5])

plays = ["run", 'pass', 'rpo']
with pm.Model() as mod:

    params = pm.Dirichlet('params', a=alphas, shape=3)

    observed = pm.Multinomial('observed', n=6, p=params, shape=3, observed=d)  
with mod:

    trace = pm.sample(700, return_inferencedata=False)
x = pm.summary(trace).iloc[:,0:4]

x.index = plays

x
tdf = pd.DataFrame(trace['params'], columns = plays)
fig, ax = plt.subplots()

tdf['run'].hist() ##blue

tdf['pass'].hist()  ##orange

tdf['rpo'].hist()  ## green
ax = pm.plot_posterior(trace, varnames = ['params'], 

                       figsize = (20, 10))

for i, a in enumerate(['run','pass','rpo']):

    ax[i].set_title(a)
ax = pm.traceplot(trace, figsize = (25, 10), combined = True)
with mod:

    ppc = pm.sample_posterior_predictive(trace)
ppc['observed'].shape
df = pd.DataFrame(ppc['observed'],index=[i for i in range(ppc['observed'].shape[0])], columns = ['run', 'pass', 'rpo'])

df
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

ax1.hist(df['run'])

ax1.title.set_text('Run')

ax2.hist(df['pass'])

ax2.title.set_text('Pass')

ax3.hist(df['rpo'])

ax3.title.set_text('RPO')
td = np.array([33, 27, 29, 49])
sig = np.sqrt(np.var(td))

mu = np.mean(td)
with pm.Model() as model:

    obs = pm.Normal('obs', mu, sig)
with model:

    trace = pm.sample(700, return_inferencedata=False)
y = pm.summary(trace).iloc[0,0:4]

y
tdf2 = pd.DataFrame(trace['obs'], columns = ['td'])

tdf2
tdf2['td'].hist()
ax = pm.traceplot(trace, figsize = (25, 10), combined = True)
ax = pm.plot_posterior(trace, varnames = ['obs'], 

                       figsize = (8, 8))