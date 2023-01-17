import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');
basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
with basic_model:
    # draw 5000 posterior samples
    trace = pm.sample(5000)
pm.traceplot(trace);
pm.summary(trace).round(2)
import seaborn as sns
plt.figure(figsize=(9,7))
sns.jointplot(trace['beta'][:,0], trace['beta'][:,1], kind="hex", color="#4CB391")
plt.xlabel("beta[0]")
plt.ylabel("beta[1]");
plt.show()

plt.figure(figsize=(9,7))
sns.jointplot(trace['alpha'], trace['sigma'], kind="hex", color="#4CB391")
plt.xlabel("alpha")
plt.ylabel("sigma");
plt.show()