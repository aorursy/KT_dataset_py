import numpy as np
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(9.5,5)})
from scipy.stats import binom
# set the parameters
param_n = 10 # number of trials
param_p = 0.3 # probability of success in a single trial
k_value = 5 #NOTE: k = the number of successes

# compute the probability/likelihood
p = binom.pmf(k=k_value, n=param_n, p=param_p) 

print('For the discrete binomial distribution:')
print(f'L(n={param_n},p={param_p}|k={k_value}) = p(k={k_value}|n={param_n},p={param_p}) = {p}')
from scipy.stats import norm
# set the parameters
param_loc = 3 # mean
param_scale = 2 # standard deviation
x_value = 5

# compute the probability/likelihood
p = norm.pdf(x=x_value, loc=param_loc,scale=param_scale)

print('For the continuous gaussian distribution:')
print(f'L(\u03BC={param_loc},\u03C3={param_scale}|x={x_value}) = f(x={x_value}|\u03BC={param_loc},\u03C3={param_scale}) = {p}')
# See https://gist.github.com/beniwohli/765262 for the unicodes for the greek alphabet 
from scipy.stats import binom

# compute the likelihoods
k_value, param_n, param_p = 21, 50, 0.5,
L1 = binom.pmf(k=k_value, n=param_n, p=param_p) 
print(f'L(n={param_n},p={param_p}|k={k_value}) = p(k={k_value}|n={param_n},p={param_p}) = {L1}')
k_value, param_n, param_p = 21, 50, 0.1 
L2 = binom.pmf(k=k_value, n=param_n, p=param_p) 
print(f'L(n={param_n},p={param_p}|k={k_value}) = p(k={k_value}|n={param_n},p={param_p}) = {L2}')
N1 = L1*0.5
print(f'Numerator for p=0.5: {N1}')
N2 = L2*0.5
print(f'Numerator for p=0.1: {N2}')
D = L1*0.5 + L2*0.5
print(f'Denominator: {D}')
print(f'Probability that the coin is fair = P(n=50,p=0.5|k=21) = {N1/D}')
print(f'Probability that the coin is not fair with p=0.1 = P(n=50,p=0.1|k=21) = {N2/D}')
# create an array of possible values for p
x = np.arange(0, 1, 0.01)
print(f'x = {x}')

# compute the likelihoods for each of these
L = binom.pmf(k=21, n=50, p=x)
print(f'L = {L}')

# compute the denominator in Bayes Theorem (i.e. the normalizing factor)
prior_prob = 1/len(L)
D = np.sum(L*prior_prob)
print(f'D = {D}')

# now compute the probability for each x-vaue using Bayes Theorem
P= L*prior_prob / D
print(f'P={P}')

import seaborn as sns
ax = sns.scatterplot(x, P)
ax.set(xlabel='x', ylabel='P(p=x)', title=f'Posterior Probability Mass Function for p (discrete distribution, every 0.01 points)');
# compute the denominator in Bayes Theorem (i.e. the normalizing factor) approximating the integral
prior_prob = 1/len(L)
delta_theta = 0.01
D = np.sum(L*prior_prob*delta_theta)
print(f'D = {D}')

# now compute the probability for each x-value using Bayes Theorem
P= L*prior_prob / D
print(f'P={P}')
ax = sns.lineplot(x, P)
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (continuous distribution)');
from scipy.stats import norm

# Create the Y,Y grid
delta_x = 0.1
delta_y = 0.1
X, Y = np.meshgrid(np.arange(2.5, 7.5, delta_x), np.arange(0.1, 2, delta_y))
# X, Y = np.meshgrid(np.arange(3, 5, 0.01), np.arange(0.1, 2, 0.01))

# compute the likelihoods for each of these

# compute the probability/likelihood
L = norm.pdf(x=4.2, loc=X, scale=Y)
print(f'L = {L}')

# compute the denominator in Bayes Theorem (i.e. the normalizing factor)
prior_prob = 1/len(L)
D = np.sum(L*(delta_x*delta_y)*prior_prob)
print(f'D = {D}')

# now compute the probability for each x-vaue using Bayes Theorem
P= L*prior_prob / D
print(f'P={P}')


Z = P
plt.contour(X, Y, Z, 20, cmap='twilight_shifted');
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar();
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='r');
ax = plt.axes(projection='3d');
ax.plot_surface(X, Y, Z, cmap='jet');
from matplotlib import cm# Normalize the colors based on Z value
norm = plt.Normalize(Z.min(), Z.max())
colors = cm.jet(norm(Z))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 55, cmap='twilight_shifted');
# create an array of possible values for p
x = np.arange(0, 1, 0.01)

# compute the likelihoods for each of these
L = binom.pmf(k=21, n=50, p=x)

# compute the denominator in Bayes Theorem (i.e. the normalizing factor) approximating the integral
prior_prob = 1/len(L)
delta_theta = 0.01
D = np.sum(L*prior_prob*delta_theta)

# now compute the probability for each x-value using Bayes Theorem
P= L*prior_prob / D

ax = sns.lineplot(x, P)
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (continuous distribution)');
from scipy.stats import norm

param_mean_for_prior = 0.5
param_stdv_for_prior = 0.01

prior_prob = norm.pdf(x=x, loc=param_mean_for_prior, scale=param_stdv_for_prior)

# plot the prior distribution
ax = sns.lineplot(x, prior_prob)
ax.set(xlabel='x', ylabel='P(p=x)', title=f'Prior Probability Distribution Expecting A Fair Coin');
D = np.sum(L*prior_prob*delta_theta)

# now compute the probability for each x-value using Bayes Theorem
P_expert_prior= L*prior_prob / D

ax = sns.lineplot(x, P)
ax = sns.lineplot(x, P_expert_prior)
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (continuous distribution)');
plt.legend(labels=['P, Flat Prior', 'P, Expert Prior']);
from scipy.stats import beta

# create our beta prior
param_prior_a = 1
param_prior_b = 20
prior_prob = beta.pdf(x=x, a=param_prior_a, b=param_prior_b)

# plot the prior distribution
ax = sns.lineplot(x, prior_prob)
ax.set(xlabel='x', ylabel='P(p=x)', title=f'Beta Prior Probability Distribution with a={param_prior_a} and b={param_prior_b}');
from scipy.stats import binom
from scipy.stats import beta

# create an array of possible values for p
x = np.arange(0, 1, 0.01)

# compute the likelihoods for each of these
param_k = 21
param_n = 50
L = binom.pmf(k=param_k, n=param_n, p=x)

# compute the denominator in Bayes Theorem (i.e. the normalizing factor) approximating the integral
delta_theta = 0.01
D_beta_prior = np.sum(L*prior_prob*delta_theta)

# now compute the probability for each x-value using Bayes Theorem
P_beta_prior= L*prior_prob / D_beta_prior

ax = sns.lineplot(x, P)
ax = sns.lineplot(x, P_beta_prior)
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (continuous distribution)');
plt.legend(labels=['P, Flat Prior', 'P, Conjugate Beta Prior']);
# create our posterior
param_posterior_a = param_k + param_prior_a
param_posterior_b = param_n - param_k + param_prior_b

# now compute the probability using the fact that the posterior probability is a beta distribution
P_beta_prior_formula= beta.pdf(x=x, a=param_posterior_a, b=param_posterior_b)

ax = sns.lineplot(x, P)
ax = sns.lineplot(x, P_beta_prior)
ax = sns.lineplot(x, P_beta_prior_formula, style=True, dashes=[(4,4)])
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (continuous distribution)');
plt.legend(labels=['P, Flat Prior', 'P, Conjugate Beta Prior', 'P, Conjugate Beta Prior (formula)']);
from scipy.stats import binom
from scipy.stats import beta

def posterior_from_conjugate_prior(**kwargs):
    if kwargs['Likelihood_Dist_Type'] == 'Binomial':
        # Get the parameters for the likelihood and prior distribution from the key word arguments.
        x = kwargs['x'] # This is state space of possible values for p = 'probability of success' in [0,1]
        n = kwargs['n'] # This is the number of Bernoili trials.
        k = kwargs['k'] # This is the number of 'successes'.
        a = kwargs['a'] # This is the parameter alpha for the prior Beta distribution
        b = kwargs['b'] # This is the parameter beta for the prior Beta distribution
        
        print(f'a_prime = {k + a}.')
        print(f'b_prime = {n - k + b}.')
        Likelihood = binom.pmf(p=x, n=n, k=k)
        Prior = beta.pdf(x=x, a=a, b=b)
        Posterior = beta.pdf(x=x, a=k+a, b=n-k+b)
        
        return [Prior, Likelihood, Posterior]
                    
    else:
        print('Distribution type not supported.')    

x = np.arange(0, 1, 0.01)
Prior, Likelihood, Posterior = posterior_from_conjugate_prior(
    Likelihood_Dist_Type='Binomial', 
    x=x, 
    n=50, 
    k=21, 
    a=1, 
    b=20)    

ax1 = sns.lineplot(x, Prior, color='red')
ax1.set(xlabel='x', ylabel='f(x)', title=f'Prior PDF');
plt.legend(labels=['Prior PDF']);
plt.show()

ax2 = sns.lineplot(x, Likelihood)
ax2.set(xlabel='x', ylabel='f(x)', title=f'Likelihood Function');
plt.legend(labels=['Likelihood Function']);
plt.show()

ax3 = sns.lineplot(x, Posterior, color='orange')
ax3.set(xlabel='x', ylabel='f(x)', title=f'Posterior PDF');
plt.legend(labels=['Posterior PDF']);
plt.show()
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import norm

def posterior_from_conjugate_prior(**kwargs):
    if kwargs['Likelihood_Dist_Type'] == 'Binomial':
        # Get the parameters for the likelihood and prior distribution from the key word arguments.
        x = kwargs['x'] # This is state space of possible values for p = 'probability of success' in [0,1]
        n = kwargs['n'] # This is the number of Bernoili trials.
        k = kwargs['k'] # This is the number of 'successes'.
        a = kwargs['a'] # This is the parameter alpha for the prior Beta distribution
        b = kwargs['b'] # This is the parameter beta for the prior Beta distribution
        
        print(f'a_prime = {k + a}.')
        print(f'b_prime = {n - k + b}.')
        Likelihood = binom.pmf(p=x, n=n, k=k)
        Prior = beta.pdf(x=x, a=a, b=b)
        Posterior = beta.pdf(x=x, a=k+a, b=n-k+b)
        
        return [Prior, Likelihood, Posterior]
    
    elif kwargs['Likelihood_Dist_Type'] == 'Gaussian_Known_Variance':
        # Get the parameters for the likelihood and prior distribution from the key word arguments.
        x = kwargs['x'] # This is state space of possible values for x in (-infinity,infinity)
        mu = kwargs['mu'] # This is the mean from the data
        var = kwargs['var'] # This is the variance from the data
        prior_mu = kwargs['prior_mu'] # This is the mean for the prior on mu
        prior_var = kwargs['prior_var'] # This is the variance for the prior on mu
        print(kwargs)
        
        # To answer the challenge question, modify this section with the correct formulas
        print(f'mu_prime = na.')
        print(f'var_prime = na.')
        Likelihood = -1
        Prior = -1
        Posterior = -1
        
        return [Prior, Likelihood, Posterior]
    
    else:
        print('Distribution type not supported.') 
        return -1, -1, -1
        
x = np.arange(0, 1, 0.01)
Prior, Likelihood, Posterior = posterior_from_conjugate_prior(Likelihood_Dist_Type='Gaussian_Known_Variance', x=x, mu=50, var=21, prior_mu=0, prior_var=1)    
