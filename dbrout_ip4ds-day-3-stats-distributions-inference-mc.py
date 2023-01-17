#START WITH RANDOM NUMBERS
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 10, 6



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy.linalg import cholesky



cov = np.array([

        [  3.40, -1.75, -1.00],

        [ -2.75,  5.50,  6.50],

        [ -2.00,  1.50,  10.25]

    ])

L = cholesky(cov)

num_samples = 1000

num_variables = 3

uncorrelated = np.random.standard_normal((num_variables, num_samples))

correlated = np.dot(L, uncorrelated) 

correlated.shape
plt.figure(figsize=(10,6))

plt.subplot(2,2,1)

plt.plot(correlated[0,:], correlated[1,:], 'b.')

plt.ylabel('dim 1')

#plt.axis('equal')

plt.grid(True)



plt.subplot(2,2,3)

plt.plot(correlated[0,:], correlated[2,:], 'b.')

#plt.plot(mu[0], mu[2], 'ro')

#plt.xlabel('y[0]')

#plt.ylabel('y[2]')

#plt.axis('equal')

plt.grid(True)



plt.subplot(2,2,4)

plt.plot(correlated[1,:], correlated[2,:], 'b.')

#plt.plot(mu[1], mu[2], 'ro')

#plt.xlabel('y[1]')

#plt.axis('equal')

plt.grid(True)


from scipy.linalg import eigh, cholesky

from scipy.stats import norm

%matplotlib inline



from pylab import plot, show, axis, subplot, xlabel, ylabel, grid





# Choice of cholesky or eigenvector method.

method = 'cholesky'

method = 'eigenvectors'



num_samples = 400



# The desired covariance matrix.

r = np.array([

        [  3.40, -2.75, -2.00],

        [ -2.75,  5.50,  1.50],

        [ -2.00,  1.50,  1.25]

    ])



# Generate samples from three independent normally distributed random

# variables (with mean 0 and std. dev. 1).

x = norm.rvs(size=(3, num_samples))



# Compute the Cholesky decomposition.

c = cholesky(r, lower=True)



# Convert the data to correlated random variables. 

y = np.dot(c, x)



#

# Plot various projections of the samples.

#

subplot(2,2,1)

plot(y[0], y[1], 'b.')

ylabel('y[1]')

axis('equal')

grid(True)



subplot(2,2,3)

plot(y[0], y[2], 'b.')

xlabel('y[0]')

ylabel('y[2]')

axis('equal')

grid(True)



subplot(2,2,4)

plot(y[1], y[2], 'b.')

xlabel('y[1]')

axis('equal')

grid(True)
# Now convert these to a polynomial model --> each dimension is a parameter
# scipy optimize/leastsqrs. errors on fitted parameters.
import numpy as np

import matplotlib.pyplot as plt
disaster_data = [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,

3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,

2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,

1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,

0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,

3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,

0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]

disaster_years = np.arange(1851,1962)
plt.figure(figsize=(10,6))

plt.bar(disaster_years,disaster_data);
!pip install corner

!pip install arviz

import pymc3



#have a look at the ways we can model here:

#https://docs.pymc.io/api/distributions
#create an empty model

with pymc3.Model() as build_model:

    

    #lets define the model parameters M and priors p(M)

    early_rate = pymc3.Exponential('early_rate',1)#bounded by zero, draw

    late_rate = pymc3.Exponential('late_rate',1)

    switch_year = pymc3.DiscreteUniform('switch_year', lower=1851, upper=1962)

    print('Defined Parameters')

    

    #we still need to connect up the model

    #here we allocate appropriate Poisson rates to years before and after current

                                                    #condition,         if True ,    else 

    disaster_model = pymc3.math.switch(switch_year >= disaster_years, early_rate, late_rate)

    print('Defined Math')



    #now we have a model disaster rate (draw), now lets compute the likelihood p(D|M)

    disasters = pymc3.Poisson('disasters', disaster_model, observed=disaster_data) 

    print('Defined Likelihood')

    

    #now we are ready to sample with mcmc!

    samples = pymc3.sample(25000)

    print('Done Sampling!')

    
print(samples['early_rate'].shape)

print(samples['late_rate'].shape)

print(samples['switch_year'].shape)

samplesdf = pymc3.trace_to_dataframe(samples)

samplesdf.head()
pymc3.traceplot(samples,var_names=('early_rate', 'late_rate'));
import corner

sigmas = np.array([1,2,3])

levels = 1.0 - np.exp(-1 * sigmas ** 2 / 2)

oneD_quantiles = (0.16, 0.84) # one sigma

fig = corner.corner(samplesdf.values, labels=samplesdf.columns, levels=levels, quantiles=oneD_quantiles);