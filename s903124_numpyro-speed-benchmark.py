!pip install numpyro
import time

import pandas as pd

import numpy as onp



import jax.numpy as np

from jax import random



import numpyro

import numpyro.distributions as dist

from numpyro.examples.datasets import COVTYPE, load_dataset

from numpyro.infer import HMC, MCMC, NUTS



from jax.random import PRNGKey



numpyro.set_platform("gpu") #Use GPU for MCMC
import pandas as pd

baseball = pd.read_csv('../input/mlb-pitch-data-20152018/atbats.csv')

d = baseball
d = d[(d.ab_id > 2018000000) & (d.ab_id < 2019000000)]

len(d)
# d = d[:200]
hit_event = ['Single','Double','Triple','Home Run']

d['Hits'] = d.event.isin(hit_event).astype(int)

d['pitcher_code'] = d['pitcher_id'].astype('category').cat.codes

d['batter_code'] = d['batter_id'].astype('category').cat.codes
# numpyro.set_host_device_count(2)



dat_list = {"hits": np.array(d.Hits),

            "pitcher": np.array(d.pitcher_code),

            "batter": np.array(d.batter_code)}



def model(pitcher, batter, hits=None, link=False):

    a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))

    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(5))

    b_bar = numpyro.sample("b_bar", dist.Normal(0, 10))

    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(5))





    a = numpyro.sample("a", dist.Normal(a_bar, sigma_a), sample_shape=(len(d['pitcher_code'].unique()),))

    b = numpyro.sample("b", dist.Normal(b_bar, sigma_b), sample_shape=(len(d['batter_code'].unique()),))



    # non-centered paramaterization

#     a = numpyro.sample('a',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(a_bar, sigma_a)), sample_shape=(len(d['pitcher_code'].unique()),))

#     b = numpyro.sample('b',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(b_bar, sigma_b)), sample_shape=(len(d['batter_code'].unique()),))



    logit_p = a[pitcher] + b[batter]

    if link:

        p = expit(logit_p)

        numpyro.sample("p", dist.Delta(p), obs=p)

    numpyro.sample("hits", dist.Binomial(logits=logit_p), obs=hits)



mcmc = MCMC(NUTS(model), 1000, 1000, num_chains=2)

mcmc.run(PRNGKey(0), np.array(d.pitcher_code), np.array(d.batter_code), hits=np.array(d.Hits), extra_fields=('potential_energy','mean_accept_prob',))

mcmc.print_summary(0.89)
np.mean( mcmc.get_extra_fields()['mean_accept_prob'])
# import pymc3 as pm

# from pymc3 import sample, Normal, HalfCauchy, Uniform

# import numpy as np
# dat_list = {"hits": np.array(d.Hits),

#             "pitcher": np.array(d.pitcher_code),

#             "batter": np.array(d.batter_code)}

# with pm.Model() as model:



#     # Priors

#     mu_a = Normal('mu_a', mu=0., tau=0.01)

#     sigma_a = HalfCauchy('sigma_a', 5)

#     mu_b = Normal('mu_b', mu=0., tau=0.01)

#     sigma_b = HalfCauchy('sigma_b', 5)





#     a = Normal('a', mu=mu_a, sigma=sigma_a, shape=len(d['pitcher_code'].unique()))



#     b = Normal('b', mu=mu_b, sigma=sigma_b, shape=len(d['batter_code'].unique()))



#     # Expected value

#     logit_p = a[dat_list['pitcher']] + b[dat_list['batter']]



#     # Data likelihood

#     p = pm.Bernoulli('y', logit_p=logit_p, observed=dat_list['hits'])
# with model:

#     trace = sample(1000, tune=1000)