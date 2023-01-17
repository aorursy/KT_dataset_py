# Dependencies



# Standard Dependencies

import os

import numpy as np

import pandas as pd

from math import sqrt



# Visualization

from pylab import *

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import seaborn as sns



# Statistics

from statistics import median

from scipy import signal

from scipy.misc import factorial

import scipy.stats as stats

from scipy.stats import sem, binom, lognorm, poisson, bernoulli, spearmanr

from scipy.fftpack import fft, fftshift



# Scikit-learn for Machine Learning models

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Seed for reproducability

seed = 12345

np.random.seed(seed)



# Kaggle Directory for Kernels

KAGGLE_DIR = '../input/'



# Read in csv of Toy Dataset

# We will use this dataset throughout the tutorial

df = pd.read_csv(KAGGLE_DIR + 'toy_dataset.csv')



# Files and file sizes

print('\n# Files and file sizes')

for file in os.listdir(KAGGLE_DIR):

    print('{}| {} MB'.format(file.ljust(30), 

                             str(round(os.path.getsize(KAGGLE_DIR + file) / 1000000, 2))))
# PMF Visualization

n = 100

p = 0.5



fig, ax = plt.subplots(1, 1, figsize=(17,5))

x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))

ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='Binomial PMF')

ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)

rv = binom(n, p)

#ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1, label='frozen PMF')

ax.legend(loc='best', frameon=False, fontsize='xx-large')

plt.title('PMF of a binomial distribution (n=100, p=0.5)', fontsize='xx-large')

plt.show()
# Plot normal distribution

mu = 0

variance = 1

sigma = sqrt(variance)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.figure(figsize=(16,5))

plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal Distribution')

plt.title('Normal Distribution with mean = 0 and std = 1')

plt.legend(fontsize='xx-large')

plt.show()
# Data

X  = np.arange(-2, 2, 0.01)

Y  = exp(-X ** 2)



# Normalize data

Y = Y / (0.01 * Y).sum()



# Plot the PDF and CDF

plt.figure(figsize=(15,5))

plt.title('Continuous Normal Distributions', fontsize='xx-large')

plot(X, Y, label='Probability Density Function (PDF)')

plot(X, np.cumsum(Y * 0.01), 'r', label='Cumulative Distribution Function (CDF)')

plt.legend(fontsize='xx-large')

plt.show()
# Uniform distribution (between 0 and 1)

uniform_dist = np.random.random(1000)

uniform_df = pd.DataFrame({'value' : uniform_dist})

uniform_dist = pd.Series(uniform_dist)
plt.figure(figsize=(18,5))

sns.scatterplot(data=uniform_df)

plt.legend(fontsize='xx-large')

plt.title('Scatterplot of a Random/Uniform Distribution', fontsize='xx-large')
plt.figure(figsize=(18,5))

sns.distplot(uniform_df)

plt.title('Random/Uniform distribution', fontsize='xx-large')
# Generate Normal Distribution

normal_dist = np.random.randn(10000)

normal_df = pd.DataFrame({'value' : normal_dist})

# Create a Pandas Series for easy sample function

normal_dist = pd.Series(normal_dist)



normal_dist2 = np.random.randn(10000)

normal_df2 = pd.DataFrame({'value' : normal_dist2})

# Create a Pandas Series for easy sample function

normal_dist2 = pd.Series(normal_dist)



normal_df_total = pd.DataFrame({'value1' : normal_dist, 

                                'value2' : normal_dist2})
# Scatterplot

plt.figure(figsize=(18,5))

sns.scatterplot(data=normal_df)

plt.legend(fontsize='xx-large')

plt.title('Scatterplot of a Normal Distribution', fontsize='xx-large')
# Normal Distribution as a Bell Curve

plt.figure(figsize=(18,5))

sns.distplot(normal_df)

plt.title('Normal distribution (n=1000)', fontsize='xx-large')
# Change of heads (outcome 1)

p = 0.6



# Create Bernoulli samples

bern_dist = bernoulli.rvs(p, size=1000)

bern_df = pd.DataFrame({'value' : bern_dist})

bern_values = bern_df['value'].value_counts()



# Plot Distribution

plt.figure(figsize=(18,4))

bern_values.plot(kind='bar', rot=0)

plt.annotate(xy=(0.85,300), 

             s='Samples that came up Tails\nn = {}'.format(bern_values[0]), 

             fontsize='large', 

             color='white')

plt.annotate(xy=(-0.2,300), 

             s='Samples that came up Heads\nn = {}'.format(bern_values[1]), 

             fontsize='large', 

             color='white')

plt.title('Bernoulli Distribution: p = 0.6, n = 1000')
x = np.arange(0, 20, 0.1)

y = np.exp(-5)*np.power(5, x)/factorial(x)



plt.figure(figsize=(15,8))

plt.title('Poisson distribution with lambda=5', fontsize='xx-large')

plt.plot(x, y, 'bs')

plt.show()
# Specify standard deviation and mean

std = 1

mean = 5



# Create log-normal distribution

dist=lognorm(std,loc=mean)

x=np.linspace(0,15,200)



# Visualize log-normal distribution

plt.figure(figsize=(15,6))

plt.xlim(5, 10)

plt.plot(x,dist.pdf(x), label='log-normal PDF')

plt.plot(x,dist.cdf(x), label='log-normal CDF')

plt.legend(fontsize='xx-large')

plt.title('Visualization of log-normal PDF and CDF', fontsize='xx-large')

plt.show()
# Summary

print('Summary Statistics for a normal distribution: ')

# Median

medi = median(normal_dist)

print('Median: ', medi)

display(normal_df.describe())



# Standard Deviation

std = sqrt(np.var(normal_dist))



print('The first four calculated moments of a normal distribution: ')

# Mean

mean = normal_dist.mean()

print('Mean: ', mean)



# Variance

var = np.var(normal_dist)

print('Variance: ', var)



# Return unbiased skew normalized by N-1

skew = normal_df['value'].skew()

print('Skewness: ', skew)



# Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis 

# (kurtosis of normal == 0.0) normalized by N-1

kurt = normal_df['value'].kurtosis()

print('Kurtosis: ', kurt)
# Take sample

normal_df_sample = normal_df.sample(100)



# Calculate Expected Value (EV), population mean and bias

ev = normal_df_sample.mean()[0]

pop_mean = normal_df.mean()[0]

bias = ev - pop_mean
print('Sample mean (Expected Value): ', ev)

print('Population mean: ', pop_mean)

print('Bias: ', bias)
from math import sqrt



Y = 100 # Actual Value

YH = 94 # Predicted Value



# MSE Formula 

def MSE(Y, YH):

     return np.square(YH - Y).mean()



# RMSE formula

def RMSE(Y, YH):

    return sqrt(np.square(YH - Y).mean())





print('MSE: ', MSE(Y, YH))



print('RMSE: ', RMSE(Y, YH))
# Standard Error (SE)

uni_sample = uniform_dist.sample(100)

norm_sample = normal_dist.sample(100)



print('Standard Error of uniform sample: ', sem(uni_sample))

print('Standard Error of normal sample: ', sem(norm_sample))



# The random samples from the normal distribution should have a higher standard error
# Note that we take very small samples just to illustrate the different sampling methods



print('---Non-Representative samples:---\n')

# Convenience samples

con_samples = normal_dist[0:5]

print('Convenience samples:\n\n{}\n'.format(con_samples))



# Haphazard samples (Picking out some numbers)

hap_samples = [normal_dist[12], normal_dist[55], normal_dist[582], normal_dist[821], normal_dist[999]]

print('Haphazard samples:\n\n{}\n'.format(hap_samples))



# Purposive samples (Pick samples for a specific purpose)

# In this example we pick the 5 highest values in our distribution

purp_samples = normal_dist.nlargest(n=5)

print('Purposive samples:\n\n{}\n'.format(purp_samples))



print('---Representative samples:---\n')



# Simple (pseudo)random sample

rand_samples = normal_dist.sample(5)

print('Random samples:\n\n{}\n'.format(rand_samples))



# Systematic sample (Every 2000th value)

sys_samples = normal_dist[normal_dist.index % 2000 == 0]

print('Systematic samples:\n\n{}\n'.format(sys_samples))



# Stratified Sampling

# We will get 1 person from every city in the dataset

# We have 8 cities so that makes a total of 8 samples

df = pd.read_csv(KAGGLE_DIR + 'toy_dataset.csv')



strat_samples = []



for city in df['City'].unique():

    samp = df[df['City'] == city].sample(1)

    strat_samples.append(samp['Income'].item())

    

print('Stratified samples:\n\n{}\n'.format(strat_samples))



# Cluster Sampling

# Make random clusters of ten people (Here with replacement)

c1 = normal_dist.sample(10)

c2 = normal_dist.sample(10)

c3 = normal_dist.sample(10)

c4 = normal_dist.sample(10)

c5 = normal_dist.sample(10)



# Take sample from every cluster (with replacement)

clusters = [c1,c2,c3,c4,c5]

cluster_samples = []

for c in clusters:

    clus_samp = c.sample(1)

    cluster_samples.extend(clus_samp)

print('Cluster samples:\n\n{}'.format(cluster_samples))    

# Covariance between Age and Income

print('Covariance between Age and Income: ')



df[['Age', 'Income']].cov()
# Correlation between two normal distributions

# Using Pearson's correlation

print('Pearson: ')

df[['Age', 'Income']].corr(method='pearson')
# Using Spearman's rho correlation

print('Spearman: ')

df[['Age', 'Income']].corr(method='spearman')
# Generate data

x = np.random.uniform(low=20, high=260, size=100)

y = 50000 + 2000*x - 4.5 * x**2 + np.random.normal(size=100, loc=0, scale=10000)



# Plot data with Linear Regression

plt.figure(figsize=(16,5))

plt.title('Well fitted but not well fitting: Linear regression plot on quadratic data', fontsize='xx-large')

sns.regplot(x, y)
# Linear regression from scratch

import random

# Create data from regression

xs = np.array(range(1,20))

ys = [0,8,10,8,15,20,26,29,38,35,40,60,50,61,70,75,80,88,96]



# Put data in dictionary

data = dict()

for i in list(xs):

    data.update({xs[i-1] : ys[i-1]})



# Slope

m = 0

# y intercept

b = 0

# Learning rate

lr = 0.0001

# Number of epochs

epochs = 100000



# Formula for linear line

def lin(x):

    return m * x + b



# Linear regression algorithm

for i in range(epochs):

    # Pick a random point and calculate vertical distance and horizontal distance

    rand_point = random.choice(list(data.items()))

    vert_dist = abs((m * rand_point[0] + b) - rand_point[1])

    hor_dist = rand_point[0]



    if (m * rand_point[0] + b) - rand_point[1] < 0:

        # Adjust line upwards

        m += lr * vert_dist * hor_dist

        b += lr * vert_dist   

    else:

        # Adjust line downwards

        m -= lr * vert_dist * hor_dist

        b -= lr * vert_dist

        

# Plot data points and regression line

plt.figure(figsize=(15,6))

plt.scatter(data.keys(), data.values())

plt.plot(xs, lin(xs))

plt.title('Linear Regression result')  

print('Slope: {}\nIntercept: {}'.format(m, b))
# scikit-learn bootstrap package

from sklearn.utils import resample



# data sample

data = df['Income']



# prepare bootstrap samples

boot = resample(data, replace=True, n_samples=5, random_state=1)

print('Bootstrap Sample: \n{}\n'.format(boot))

print('Mean of the population: ', data.mean())

print('Standard Deviation of the population: ', data.std())



# Bootstrap plot

pd.plotting.bootstrap_plot(data)
# Perform t-test and compute p value of two random samples

print('T-statistics and p-values of two random samples.')

for _ in range(10):

    rand_sample1 = np.random.random_sample(10)

    rand_sample2 = np.random.random_sample(10)

    print(stats.ttest_ind(rand_sample1, rand_sample2))
# To-do

# Equivalence testing
# q-q plot of a normal distribution

plt.figure(figsize=(15,6))

stats.probplot(normal_dist, dist="norm", plot=plt)

plt.show()
# q-q plot of a uniform/random distribution

plt.figure(figsize=(15,6))

stats.probplot(uniform_dist, dist="norm", plot=plt) 

plt.show()
# Detect outliers on the 'Income' column of the Toy Dataset



# Function for detecting outliers a la Tukey's method using z-scores

def tukey_outliers(data) -> list:

    # For more information on calculating the threshold check out:

    # https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32

    threshold = 3

    

    mean = np.mean(data)

    std = np.std(data)

    

    # Spot and collect outliers

    outliers = []

    for i in data:

        z_score = (i - mean) / std

        if abs(z_score) > threshold:

            outliers.append(i)

    return outliers



# Get outliers

income_outliers = tukey_outliers(df['Income'])



# Visualize distribution and outliers

plt.figure(figsize=(15,6))

df['Income'].plot(kind='hist', bins=100, label='Income distribution')

plt.hist(income_outliers, bins=20, label='Outliers')

plt.title("Outlier detection a la Tukey's method", fontsize='xx-large')

plt.xlabel('Income')

plt.legend(fontsize='xx-large')
# Inverse logit function (link function)

def inv_logit(x):

    return 1 / (1 + np.exp(-x))



t1 = np.arange(-10, 10, 0.1)

plt.figure(figsize=(15,6))

plt.plot(t1, inv_logit(t1), 

         t1, inv_logit(t1-2),   

         t1, inv_logit(t1*2))

plt.title('Inverse logit functions', fontsize='xx-large')

plt.legend(('Normal', 'Changed intercept', 'Changed slope'), fontsize='xx-large')
# Simple example of Logistic Regression in Python

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)



# Logistic regression classifier

clf = LogisticRegression(random_state=0, 

                         solver='lbfgs',

                         multi_class='multinomial').fit(X, y)



print('Accuracy score of logistic regression model on the Iris flower dataset: {}'.format(clf.score(X, y)))
