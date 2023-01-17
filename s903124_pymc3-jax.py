!pip install -U --no-deps git+https://github.com/pymc-devs/Theano-PyMC.git

!pip uninstall typing -y

!pip install -U --no-deps git+https://github.com/pymc-devs/pymc3.git@pymc3jax

!pip install tfp-nightly

!pip install xarray==0.16.0 -U

!pip install jax

!pip install jaxlib

!pip install numpyro
import arviz as az

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pymc3 as pm

import theano

import pymc3.sampling_jax

import time

print('Running on PyMC3 v{}'.format(pm.__version__))
%config InlineBackend.figure_format = 'retina'

az.style.use('arviz-darkgrid')
data = pd.read_csv(pm.get_data('radon.csv'))

data = pd.concat([data], ignore_index=True)

data['log_radon'] = data['log_radon'].astype(theano.config.floatX)

county_names = data.county.unique()

county_idx = data.county_code.values.astype('int32')



n_counties = len(data.county.unique())
with pm.Model() as hierarchical_model:

    # Hyperpriors for group nodes

    mu_a = pm.Normal('mu_a', mu=0., sigma=100.)

    sigma_a = pm.HalfNormal('sigma_a', 5.)

    mu_b = pm.Normal('mu_b', mu=0., sigma=100.)

    sigma_b = pm.HalfNormal('sigma_b', 5.)



    # Intercept for each county, distributed around group mean mu_a

    # Above we just set mu and sd to a fixed value while here we

    # plug in a common group distribution for all a and b (which are

    # vectors of length n_counties).

    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_counties)

    # Intercept for each county, distributed around group mean mu_a

    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_counties)



    # Model error

    eps = pm.HalfCauchy('eps', 5.)



    radon_est = a[county_idx] + b[county_idx]*data.floor.values



    # Data likelihood

    radon_like = pm.Normal('radon_like', mu=radon_est,

                           sigma=eps, observed=data.log_radon)



%%time

# Inference button (TM)!

with hierarchical_model:

    hierarchical_trace_jax = pm.sampling_jax.sample_numpyro_nuts(

        2000, tune=2000, target_accept=.9)



%%time

with hierarchical_model:

    hierarchical_trace = pm.sample(2000, tune=2000, target_accept=.9, 

                                   compute_convergence_checks=False)
pm.traceplot(hierarchical_trace_jax, 

             var_names=['mu_a', 'mu_b',

                        'sigma_a_log__', 'sigma_b_log__',

                        'eps_log__']);
pm.traceplot(hierarchical_trace_jax, 

             var_names=['a'], coords={'a_dim_0': range(5)});