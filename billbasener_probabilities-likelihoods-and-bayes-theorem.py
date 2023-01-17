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
# import uniform distribution
from scipy.stats import uniform

# generate random numbers from a uniform distribution
sample_size = 10000
param_loc = 5
param_scale = 10
data_uniform = uniform.rvs(size=sample_size, loc=param_loc, scale=param_scale)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_uniform[0:5])

# plot a historgram of the output
ax = sns.distplot(data_uniform,
                  bins=100,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Uniform Distribution: Sample Size = {sample_size}. loc={param_loc}, scale={param_scale}');
from scipy.stats import norm

# generate random numbers from a normal distribution

sample_size = 10000
param_loc = 3 # mean
param_scale = 2 # standard deviation
data_normal = norm.rvs(size=sample_size,loc=param_loc,scale=param_scale)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_normal[0:5])

# plot a histogram of the output
ax = sns.distplot(data_normal,
                  bins=100,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Normal Distribution: Sample Size = {sample_size}, loc={param_loc}, scale={param_scale}');
# import bernoulli
from scipy.stats import bernoulli

# generate bernoulli data
sample_size = 100000
param_p = 0.3
data_bern = bernoulli.rvs(size=sample_size,p=param_p)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_bern[0:5])

# Create the Plot
ax= sns.distplot(data_bern,                  
                  kde=False,
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Bernoulli Distribution: Sample Size = {sample_size}, p={param_p}');
ax.legend();
from scipy.stats import binom

# Generate Binomial Data
sample_size = 10000
param_n = 10
param_p = 0.7
data_binom = binom.rvs(size=sample_size, n=param_n,p=param_p,)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_binom[0:5])

# Create the Plot
ax = sns.distplot(data_binom,
                  kde=False,
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Binomial Distribution: n={param_n} ,p={param_p}')
ax.legend();
from scipy.stats import poisson

# Generate Poisson Data
sample_size = 10000
param_mu = 3 #(often denoted lambda)
data_poisson = poisson.rvs(size=sample_size, mu=param_mu)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_poisson[0:5])

# Create the Plot
ax = sns.distplot(data_poisson,
                  kde=False,
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Poisson Distribution: Sample Size = {sample_size}, mu={param_mu}');
ax.legend();
from scipy.stats import beta

# Generate Poisson Data
sample_size = 100000
param_a = 1
param_b = 1
data_beta = beta.rvs(param_a, param_b, size=sample_size)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_beta[0:5])

# Create the Plot
ax = sns.distplot(data_beta,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Beta({param_a},{param_b}) Distribution: Sample Size = {sample_size}');
ax.legend();
from scipy.stats import beta

# Generate Poisson Data
sample_size = 100000
param_a = .5
param_b = .5
data_beta = beta.rvs(param_a, param_b, size=sample_size)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_beta[0:5])

# Create the Plot
ax = sns.distplot(data_beta,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Beta({param_a},{param_b}) Distribution: Sample Size = {sample_size}');
ax.legend();
from scipy.stats import beta

# Generate Poisson Data
sample_size = 100000
param_a = 5
param_b = 10
data_beta = beta.rvs(param_a, param_b, size=sample_size)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_beta[0:5])

# Create the Plot
ax = sns.distplot(data_beta,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Beta({param_a},{param_b}) Distribution: Sample Size = {sample_size}');
ax.legend();
from scipy.stats import gamma

# Generate Gamma Data
sample_size = 100000
param_a = 3 # shape parameter, sometimes denoted k or alpha
param_scale = 2 # this is the scale parameter theta.  Sometime this is given as rate parameter called beta, where theta=1/beta.
data_gamma = gamma.rvs(size=sample_size, a=param_a, scale=param_scale)

# print a few values from the distribution:
print('The first 5 values from this distribution:')
print(data_gamma[0:5])

# Create the Plot
ax = sns.distplot(data_gamma,
                  kde_kws={"label": "KDE"},
                  hist_kws={"label": "Histogram"})
ax.set(xlabel='x ', ylabel='Frequency', title=f'Gamma Distribution: Sample Size = {sample_size}, a=k={param_a}, scale='+r'$\theta$'+f'={param_scale}');
ax.legend();
print('Comparing the data mean to the distribution mean:')
print(np.mean(data_gamma))
print(param_a*param_scale)
# Generate Mulitvariate Gaussian Data
sample_size=10000
param_mean = [0, 2]
param_cov = [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(param_mean, param_cov, size=sample_size)
# create a data frame from the data
df = pd.DataFrame(data, columns=["x", "y"])

# Create the Plot
ax = sns.jointplot(x="x", y="y", data=df);
ax.fig.subplots_adjust(top=0.9)
ax.fig.suptitle("Scatterplot of Guassian Distribution Data");

ax = sns.jointplot(x="x", y="y", data=df, kind="kde");
ax.fig.subplots_adjust(top=0.9)
ax.fig.suptitle("Kenel Density Estimation of Gausssian Distribution Data");

ax = sns.jointplot(x="x", y="y", data=df, kind="hex", color="k");
ax.fig.subplots_adjust(top=0.9)
ax.fig.suptitle("Hexbin Plot of Guassian Distribution Data");

f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
ax = sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
ax.set(xlabel='x ', ylabel='y', title=f'Continuous Shading Plot of Gaussian Distribution Data');
# Generate Mulitvariate Gaussian Data
sample_size=1000
param_mean = [0, 2]
param_cov = [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(param_mean, param_cov, size=sample_size)
# create a data frame from the data
df = pd.DataFrame(data, columns=["x", "y"])

# Create the Plot
ax = sns.jointplot(x="x", y="y", data=df, alpha=0.3).plot_joint(sns.kdeplot, zorder=0, n_levels=6)
print('Compute the eigenvalues and eigenvectors of the covariance to compare to the plot.')
e = np.linalg.eig(param_cov)
print(f'Eigenvalues{e[0]}')
print(f'Eigenvectors{e[1]}')
from scipy.stats import wishart

# Generate data
param_df = 2
param_scale = np.asarray([[2,1],[1,2]])
data_wishart = wishart.rvs(param_df, param_scale, size=3)

# Print the data
print(data_wishart)
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
N = L1*0.5
print(f'Numerator: {N}')
D = L1*0.5 + L2*0.5
print(f'Denominator: {D}')
print(f'P(n=50,p=0.5|k=21) = {N/D}')
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
# Create the Y,Y grid
X, Y = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(0, 10, 0.1))

# THIS IS JUST A PLACEHOLDER FUNCTION FOR Z.  TO ANSWER THE CHALLENGE QUESTION, YOU MUST REPLACE THIS FORMULA WITH P(x,y).
Z = np.exp(-X**2/50-(Y-8)**2/20)
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