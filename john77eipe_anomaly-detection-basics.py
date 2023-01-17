# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat

import matplotlib.pyplot as plt

from matplotlib import cm # Colormaps

import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns



sns.set_style('darkgrid')

np.random.seed(42)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def univariate_normal(x, mean, variance):

    """pdf of the univariate normal distribution."""

    return ((1. / np.sqrt(2 * np.pi * variance)) * 

            np.exp(-(x - mean)**2 / (2 * variance)))
# Plot different Univariate Normals

x = np.linspace(-3, 5, num=150)

fig = plt.figure(figsize=(5, 3))

plt.plot(

    x, univariate_normal(x, mean=0, variance=1), 

    label="$\mathcal{N}(0, 1)$")

plt.plot(

    x, univariate_normal(x, mean=2, variance=3), 

    label="$\mathcal{N}(2, 3)$")

plt.plot(

    x, univariate_normal(x, mean=0, variance=0.2), 

    label="$\mathcal{N}(0, 0.2)$")

plt.xlabel('$x$', fontsize=13)

plt.ylabel('density: $p(x)$', fontsize=13)

plt.title('Univariate normal distributions')

plt.ylim([0, 1])

plt.xlim([-3, 5])

plt.legend(loc=1)

fig.subplots_adjust(bottom=0.15)

plt.show()

#
def multivariate_normal(x, d, mean, covariance):

    """pdf of the multivariate normal distribution."""

    x_m = x - mean

    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 

            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
# Plot bivariate distribution

def generate_surface(mean, covariance, d):

    """Helper function to generate density surface."""

    nb_of_x = 100 # grid size

    x1s = np.linspace(-5, 5, num=nb_of_x)

    x2s = np.linspace(-5, 5, num=nb_of_x)

    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid

    pdf = np.zeros((nb_of_x, nb_of_x))

    # Fill the cost matrix for each combination of weights

    for i in range(nb_of_x):

        for j in range(nb_of_x):

            pdf[i,j] = multivariate_normal(

                np.matrix([[x1[i,j]], [x2[i,j]]]), 

                d, mean, covariance)

    return x1, x2, pdf  # x1, x2, pdf(x1,x2)



# subplot

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

d = 2  # number of dimensions



# Plot of independent Normals

bivariate_mean = np.matrix([[0.], [0.]])  # Mean

bivariate_covariance = np.matrix([

    [1., 0.], 

    [0., 1.]])  # Covariance

x1, x2, p = generate_surface(

    bivariate_mean, bivariate_covariance, d)

# Plot bivariate distribution

con = ax1.contourf(x1, x2, p, 100, cmap=cm.YlGnBu)

ax1.set_xlabel('$x_1$', fontsize=13)

ax1.set_ylabel('$x_2$', fontsize=13)

ax1.axis([-2.5, 2.5, -2.5, 2.5])

ax1.set_aspect('equal')

ax1.set_title('Independent variables', fontsize=12)



# Plot of correlated Normals

bivariate_mean = np.matrix([[0.], [1.]])  # Mean

bivariate_covariance = np.matrix([

    [1., 0.8], 

    [0.8, 1.]])  # Covariance

x1, x2, p = generate_surface(

    bivariate_mean, bivariate_covariance, d)

# Plot bivariate distribution

con = ax2.contourf(x1, x2, p, 100, cmap=cm.YlGnBu)

ax2.set_xlabel('$x_1$', fontsize=13)

ax2.set_ylabel('$x_2$', fontsize=13)

ax2.axis([-2.5, 2.5, -1.5, 3.5])

ax2.set_aspect('equal')

ax2.set_title('Correlated variables', fontsize=12)



# Add colorbar and title

fig.subplots_adjust(right=0.8)

cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

cbar = fig.colorbar(con, cax=cbar_ax)

cbar.ax.set_ylabel('$p(x_1, x_2)$', fontsize=13)

plt.suptitle('Bivariate normal distributions', fontsize=13, y=0.95)

plt.show()

#