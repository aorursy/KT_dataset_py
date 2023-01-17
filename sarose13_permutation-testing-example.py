import numpy as np
import pymc3 as pm
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm

from tqdm import trange

import matplotlib.pyplot as graph
import seaborn as sns

graph.style.use('fivethirtyeight')
n = 15
m = 1
x = stats.norm(45, 10).rvs(size=(n, 1))
y = (m * x) + stats.norm(0, 10).rvs(size=(n, 1))
y = y.flatten()

print(y.shape, x.shape)

graph.plot(x, y, '.')
graph.xlabel('X')
graph.ylabel('Y')
graph.show()
display(sm.OLS(y, sm.add_constant(x)).fit().summary())
def mse(params, xi, yi):
    mi, bi = params
    return ((yi - (mi * xi + bi).flatten()) ** 2).sum()
ols_params = opt.fmin(mse, [-1., -1.], args=(x, y))
print(ols_params)
# Permuation testing
n_perm = int(10e3)
null_dist = np.zeros((n_perm, 2))
for i in trange(n_perm, desc='Permutation Testing'):
    ols_perm_params = opt.fmin(mse, [0., 0.], args=(x, np.random.choice(y, size=len(y), replace=False)), disp=False)
    null_dist[i, 0] = ols_perm_params[0]
    null_dist[i, 1] = ols_perm_params[1]
    
graph.title('Null Distribution of m')
sns.distplot(null_dist[:, 0])
graph.show()

graph.title('Null Distribution of b')
sns.distplot(null_dist[:, 1])
graph.show()
# Compute p-values
for i, name in enumerate(['m', 'b']):
    print(f'{name} p-value =', (null_dist[:, i] >= ols_params[i]).mean())
with pm.Model() as bayes_model:
    # Priors
    slope = pm.Normal('m', mu=0, sd=100**2)
    intercept = pm.Normal('b', mu=0, sd=100**2)
    resid_var = pm.HalfNormal('sd', sd=100**2)
    
    # Likelihood
    obs = pm.Normal('y', mu=slope * x.flatten() + intercept, sd=resid_var, observed=y)
    
    # Sample
    trace = pm.sample(5000)
    pm.traceplot(trace)
    graph.show()
pm.plot_posterior(trace, varnames=['m', 'b'], ref_val=[m, 0])
graph.show()

pm.plot_posterior(trace, varnames=['m', 'b'], ref_val=[0, 0])
graph.show()