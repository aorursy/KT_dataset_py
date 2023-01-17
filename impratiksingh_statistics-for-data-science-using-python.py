# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Standard Imports

import numpy as np

import pandas as pd

import matplotlib

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

%matplotlib inline





from IPython import display

from ipywidgets import interact , widgets





import scipy.stats

import scipy.optimize

import scipy.spatial



import re

import mailbox

import csv

import math

import io
# 2008 US swing state election results

df_swing=pd.read_csv('../input/2008_swing_states.csv')

df_all_states=pd.read_csv('../input/2008_all_states.csv')

df_swing[['state','county','dem_share']].head()
_ = plt.hist(df_swing['dem_share'])

# df_swing['dem_share'].plot.hist() # Same result
_ = plt.hist(df_swing['dem_share'],bins=20)
# All data is shown in beeswarmplot , but not good for large data.

_ = sns.swarmplot(x='state',y='dem_share',data=df_swing)
def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    n = len(data)



    # x-data for the ECDF: x

    x = np.sort(data)



    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n



    return x, y
# Making an ECDF

x=np.sort(df_swing['dem_share'])

y=np.arange(1,len(x)+1)/len(x)



_=plt.plot(x,y,marker='.',linestyle='none')
# Comparison of ECDFs

# ECDFs also allow you to compare two or more distributions (though plots get cluttered if you have too many). 



#pa_dem_share,oh_dem_share,fl_dem_share

pa_dem_share=df_swing.query("state=='PA'")['dem_share']

oh_dem_share=df_swing.query("state=='OH'")['dem_share']

fl_dem_share=df_swing.query("state=='FL'")['dem_share']





x_pa,y_pa=ecdf(pa_dem_share)

x_oh,y_oh=ecdf(oh_dem_share)

x_fl,y_fl=ecdf(fl_dem_share)



_=plt.plot(x_pa,y_pa,marker='.',linestyle='none')

_=plt.plot(x_oh,y_oh,marker='.',linestyle='none')

_=plt.plot(x_fl,y_fl,marker='.',linestyle='none')

plt.legend(('PA','OH','FL'))

plt.show()

# Computing means

# The mean of all measurements gives an indication of the typical magnitude of a measurement. It is computed using np.mean().

# Mean is affected by outliers but not median.



# Percentiles, outliers, and box plots

# Specify array of percentiles: percentiles

percentiles=np.array([2.5,25,50,75,97.5])



# Compute percentiles: ptiles_vers

ptiles_vers=np.percentile(df_swing['dem_share'],percentiles)

# Plot the ECDF

_ = plt.plot(x, y, '.')

_ = plt.xlabel('Dem Share')

_ = plt.ylabel('ECDF')



# Overlay percentiles as red diamonds.

_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',

         linestyle='none')



# Show the plot

plt.show()

_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)

_ = plt.xlabel('region')

_ = plt.ylabel('percent of vote for Obama')
print(np.var(x_fl))

print(np.sqrt(np.var(x_fl)))

print(np.std(x_fl))

total_votes=df_swing.total_votes/1000

dem_share=df_swing.dem_share

_ = plt.plot(total_votes, dem_share, marker='.', linestyle='none')

_ = plt.xlabel('total votes (thousands)')

_ = plt.ylabel('percent of vote for Obama')



# Indeed, we see some correlation
# Compute the covariance matrix: covariance_matrix

covariance_matrix=np.cov(total_votes,dem_share)



# Print covariance matrix

print(covariance_matrix)

def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix: corr_mat

    corr_mat=np.corrcoef(x,y)



    # Return entry [0,1]

    return corr_mat[0,1]



# Compute Pearson correlation coefficient 

r=pearson_r(total_votes,dem_share)



# Print the result

print(r)
# we'll generate lots of random numbers between zero and one, and then plot a histogram of the results.

# If the numbers are truly random, all bars in the histogram should be of (close to) equal height.



random_numbers=np.random.random(1000)

_ = plt.hist(random_numbers)





# The histogram is almost exactly flat across the top,indicating that 

# there is equal chance that a randomly-generated number is in any of the bins of the histogram.
def perform_bernoulli_trials(n, p):

    """Perform n Bernoulli trials with success probability p

    and return number of successes."""

    # Initialize number of successes: n_success

    n_success = 0



    # Perform trials

    for i in range(n):

        # Choose random number between zero and one: random_number

        random_number=np.random.random()



        # If less than p, it's a success so add one to n_success

        if random_number < p:

            n_success+=1



    return n_success
# Seed random number generator

np.random.seed(42)



# Initialize the number of defaults: n_defaults

n_defaults=np.empty(1000)



# Compute the number of defaults

for i in range(1000):

    n_defaults[i] = perform_bernoulli_trials(100,0.05)





# Plot the histogram with default number of bins; label your axes

_ = plt.hist(n_defaults, density=True)

_ = plt.xlabel('number of defaults out of 100 loans')

_ = plt.ylabel('probability')



# Show the plot

plt.show()



# This is actually not an optimal way to plot a histogram when the results are known to be integers.
# Compute ECDF: x, y

x,y=ecdf(n_defaults)



# Plot the ECDF with labeled axes

plt.plot(x,y,marker='.',linestyle='none')





plt.xlabel('number of defaults out of 100')

plt.ylabel('ECDF')

# Show the plot

plt.show()



# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money

n_lose_money=np.sum(n_defaults >=10)



# Compute and print probability of losing money

print('Probability of losing money =', n_lose_money / len(n_defaults))



# As we might expect, we most likely get 5/100 defaults.But we still have about a 2% chance of getting 10 or more defaults out of 100 loans.
# Take 10,000 samples out of the binomial distribution: n_defaults

n_defaults=np.random.binomial(n=100,p=0.05,size=10000)



# Compute CDF: x, y

x,y=ecdf(n_defaults)



# Plot the CDF with axis labels

plt.plot(x,y,marker='.',linestyle='none')

plt.xlabel('number of defaults out of 100 loans')

plt.ylabel('CDF')







# Show the plot

plt.show()

# Compute bin edges: bins

bins = np.arange(max(n_defaults) + 1.5) - 0.5



# Generate histogram

plt.hist(n_defaults,density=True,bins=bins)



# Label axes

plt.xlabel('number of defaults out of 100 loans')

plt.ylabel('PMF')





# Show the plot

plt.show()

'''

The means are all about the same, which can be shown to be true by doing some pen-and-paper work.

The standard deviation of the Binomial distribution gets closer and closer to that of the Poisson distribution

as the probability p gets lower and lower.



'''



# Draw 10,000 samples out of Poisson distribution: samples_poisson

samples_poisson=np.random.poisson(10,size=10000)



# Print the mean and standard deviation

print('Poisson:     ', np.mean(samples_poisson),

                       np.std(samples_poisson))



# Specify values of n and p to consider for Binomial: n, p

n = [20, 100, 1000]

p = [0.5, 0.1, 0.01]





# Draw 10,000 samples for each n,p pair: samples_binomial

for i in range(3):

    samples_binomial = np.random.binomial(n[i],p[i],size=10000)



    # Print results

    print('n =', n[i], 'Binom:', np.mean(samples_binomial),

                                 np.std(samples_binomial))

'''

We can see how the different standard deviations result in PDFs of different widths. The peaks are all centered at the mean of 20.

'''

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10

samples_std1=np.random.normal(20,1,size=100000)

samples_std3=np.random.normal(20,3,size=100000)

samples_std10=np.random.normal(20,10,size=100000)



# Make histograms

_=plt.hist(samples_std1,density=True,histtype='step',bins=100)

_=plt.hist(samples_std3,density=True,histtype='step',bins=100)

_=plt.hist(samples_std10,density=True,histtype='step',bins=100)



# Make a legend, set limits and show plot

_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))

plt.ylim(-0.01, 0.42)

plt.show()

'''

The CDFs all pass through the mean at the 50th percentile; the mean and median of a Normal distribution are equal.

The width of the CDF varies with the standard deviation.



'''

# Generate CDFs

x_std1,y_std1=ecdf(samples_std1)

x_std3,y_std3=ecdf(samples_std3)

x_std10,y_std10=ecdf(samples_std10)





# Plot CDFs

_=plt.plot(x_std1,y_std1,marker='.',linestyle='none')

_=plt.plot(x_std3,y_std3,marker='.',linestyle='none')

_=plt.plot(x_std10,y_std10,marker='.',linestyle='none')





# Make a legend and show the plot

_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')

plt.show()

nohitter_times=[ 843, 1613, 1101,  215,  684,  814,  278,  324,  161,  219,  545,

        715,  966,  624,   29,  450,  107,   20,   91, 1325,  124, 1468,

        104, 1309,  429,   62, 1878, 1104,  123,  251,   93,  188,  983,

        166,   96,  702,   23,  524,   26,  299,   59,   39,   12,    2,

        308, 1114,  813,  887,  645, 2088,   42, 2090,   11,  886, 1665,

       1084, 2900, 2432,  750, 4021, 1070, 1765, 1322,   26,  548, 1525,

         77, 2181, 2752,  127, 2147,  211,   41, 1575,  151,  479,  697,

        557, 2267,  542,  392,   73,  603,  233,  255,  528,  397, 1529,

       1023, 1194,  462,  583,   37,  943,  996,  480, 1497,  717,  224,

        219, 1531,  498,   44,  288,  267,  600,   52,  269, 1086,  386,

        176, 2199,  216,   54,  675, 1243,  463,  650,  171,  327,  110,

        774,  509,    8,  197,  136,   12, 1124,   64,  380,  811,  232,

        192,  731,  715,  226,  605,  539, 1491,  323,  240,  179,  702,

        156,   82, 1397,  354,  778,  603, 1001,  385,  986,  203,  149,

        576,  445,  180, 1403,  252,  675, 1351, 2983, 1568,   45,  899,

       3260, 1025,   31,  100, 2055, 4043,   79,  238, 3931, 2351,  595,

        110,  215,    0,  563,  206,  660,  242,  577,  179,  157,  192,

        192, 1848,  792, 1693,   55,  388,  225, 1134, 1172, 1555,   31,

       1582, 1044,  378, 1687, 2915,  280,  765, 2819,  511, 1521,  745,

       2491,  580, 2072, 6450,  578,  745, 1075, 1103, 1549, 1520,  138,

       1202,  296,  277,  351,  391,  950,  459,   62, 1056, 1128,  139,

        420,   87,   71,  814,  603, 1349,  162, 1027,  783,  326,  101,

        876,  381,  905,  156,  419,  239,  119,  129,  467]
# Seed random number generator

np.random.seed(42)



'''

We see the typical shape of the Exponential distribution, going from a maximum at 0 and decaying to the right.

'''

# Compute mean no-hitter time: tau

tau = np.mean(nohitter_times)



# Draw out of an exponential distribution with parameter tau: inter_nohitter_time

inter_nohitter_time = np.random.exponential(tau, 100000)



# Plot the PDF and label axes

_ = plt.hist(inter_nohitter_time,

             bins=50, density=True,histtype='step')

_ = plt.xlabel('Games between no-hitters')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
'''

It looks like no-hitters in the modern era of Major League Baseball are Exponentially distributed.

Based on the story of the Exponential distribution, this suggests that they are a random process; 

when a no-hitter will happen is independent of when the last no-hitter was.

'''



# Create an ECDF from real data: x, y

x, y = ecdf(nohitter_times)



# Create a CDF from theoretical samples: x_theor, y_theor

x_theor, y_theor = ecdf(inter_nohitter_time)



# Overlay the plots

plt.plot(x_theor, y_theor)

plt.plot(x, y, marker='.', linestyle='none')



# Margins and axis labels

plt.margins(.02)

plt.xlabel('Games between no-hitters')

plt.ylabel('CDF')



# Show the plot

plt.show()

'''

Notice how the value of tau given by the mean matches the data best. In this way, tau is an optimal parameter.

'''

# Plot the theoretical CDFs

plt.plot(x_theor, y_theor)

plt.plot(x, y, marker='.', linestyle='none')

plt.margins(0.02)

plt.xlabel('Games between no-hitters')

plt.ylabel('CDF')



# Take samples with half tau: samples_half

samples_half = np.random.exponential(tau/2,10000)



# Take samples with double tau: samples_double

samples_double = np.random.exponential(tau*2,10000)



# Generate CDFs from these samples

x_half, y_half = ecdf(samples_half)

x_double, y_double = ecdf(samples_double)



# Plot these CDFs as lines

_ = plt.plot(x_half, y_half)

_ = plt.plot(x_double, y_double)



# Show the plot

plt.show()
illiteracy=np.asarray([ 9.5, 49.2,  1. , 11.2,  9.8, 60. , 50.2, 51.2,  0.6,  1. ,  8.5,

        6.1,  9.8,  1. , 42.2, 77.2, 18.7, 22.8,  8.5, 43.9,  1. ,  1. ,

        1.5, 10.8, 11.9,  3.4,  0.4,  3.1,  6.6, 33.7, 40.4,  2.3, 17.2,

        0.7, 36.1,  1. , 33.2, 55.9, 30.8, 87.4, 15.4, 54.6,  5.1,  1.1,

       10.2, 19.8,  0. , 40.7, 57.2, 59.9,  3.1, 55.7, 22.8, 10.9, 34.7,

       32.2, 43. ,  1.3,  1. ,  0.5, 78.4, 34.2, 84.9, 29.1, 31.3, 18.3,

       81.8, 39. , 11.2, 67. ,  4.1,  0.2, 78.1,  1. ,  7.1,  1. , 29. ,

        1.1, 11.7, 73.6, 33.9, 14. ,  0.3,  1. ,  0.8, 71.9, 40.1,  1. ,

        2.1,  3.8, 16.5,  4.1,  0.5, 44.4, 46.3, 18.7,  6.5, 36.8, 18.6,

       11.1, 22.1, 71.1,  1. ,  0. ,  0.9,  0.7, 45.5,  8.4,  0. ,  3.8,

        8.5,  2. ,  1. , 58.9,  0.3,  1. , 14. , 47. ,  4.1,  2.2,  7.2,

        0.3,  1.5, 50.5,  1.3,  0.6, 19.1,  6.9,  9.2,  2.2,  0.2, 12.3,

        4.9,  4.6,  0.3, 16.5, 65.7, 63.5, 16.8,  0.2,  1.8,  9.6, 15.2,

       14.4,  3.3, 10.6, 61.3, 10.9, 32.2,  9.3, 11.6, 20.7,  6.5,  6.7,

        3.5,  1. ,  1.6, 20.5,  1.5, 16.7,  2. ,  0.9])



fertility=np.asarray([1.769, 2.682, 2.077, 2.132, 1.827, 3.872, 2.288, 5.173, 1.393,

       1.262, 2.156, 3.026, 2.033, 1.324, 2.816, 5.211, 2.1  , 1.781,

       1.822, 5.908, 1.881, 1.852, 1.39 , 2.281, 2.505, 1.224, 1.361,

       1.468, 2.404, 5.52 , 4.058, 2.223, 4.859, 1.267, 2.342, 1.579,

       6.254, 2.334, 3.961, 6.505, 2.53 , 2.823, 2.498, 2.248, 2.508,

       3.04 , 1.854, 4.22 , 5.1  , 4.967, 1.325, 4.514, 3.173, 2.308,

       4.62 , 4.541, 5.637, 1.926, 1.747, 2.294, 5.841, 5.455, 7.069,

       2.859, 4.018, 2.513, 5.405, 5.737, 3.363, 4.89 , 1.385, 1.505,

       6.081, 1.784, 1.378, 1.45 , 1.841, 1.37 , 2.612, 5.329, 5.33 ,

       3.371, 1.281, 1.871, 2.153, 5.378, 4.45 , 1.46 , 1.436, 1.612,

       3.19 , 2.752, 3.35 , 4.01 , 4.166, 2.642, 2.977, 3.415, 2.295,

       3.019, 2.683, 5.165, 1.849, 1.836, 2.518, 2.43 , 4.528, 1.263,

       1.885, 1.943, 1.899, 1.442, 1.953, 4.697, 1.582, 2.025, 1.841,

       5.011, 1.212, 1.502, 2.516, 1.367, 2.089, 4.388, 1.854, 1.748,

       2.978, 2.152, 2.362, 1.988, 1.426, 3.29 , 3.264, 1.436, 1.393,

       2.822, 4.969, 5.659, 3.24 , 1.693, 1.647, 2.36 , 1.792, 3.45 ,

       1.516, 2.233, 2.563, 5.283, 3.885, 0.966, 2.373, 2.663, 1.251,

       2.052, 3.371, 2.093, 2.   , 3.883, 3.852, 3.718, 1.732, 3.928])
'''

You can see the correlation between illiteracy and fertility by eye, and by the substantial Pearson correlation coefficient of 0.8.

It is difficult to resolve in the scatter plot, but there are many points around near-zero illiteracy and about 1.8 children/woman.

'''

# Plot the illiteracy rate versus fertility

_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')



# Set the margins and label axes

plt.margins(.02)

_ = plt.xlabel('percent illiterate')

_ = plt.ylabel('fertility')



# Show the plot

plt.show()



# Show the Pearson correlation coefficient

print(pearson_r(illiteracy, fertility))

# Plot the illiteracy rate versus fertility

_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

plt.margins(0.02)

_ = plt.xlabel('percent illiterate')

_ = plt.ylabel('fertility')



# Perform a linear regression using np.polyfit(): a, b

a, b = np.polyfit(illiteracy,fertility ,deg=1)



# Print the results to the screen

print('slope =', a, 'children per woman / percent illiterate')

print('intercept =', b, 'children per woman')



# Make theoretical line to plot

x = np.array([0,100])

y = a * x + b



# Add regression line to your plot

_ = plt.plot(x, y)



# Draw the plot

plt.show()

'''

Notice that the minimum on the plot, that is the value of the slope that gives the minimum sum of the square of the residuals,

is the same value you got when performing the regression.

'''

# Specify slopes to consider: a_vals

a_vals = np.linspace(0,0.1,200)



# Initialize sum of square of residuals: rss

rss = np.empty_like(a_vals)



# Compute sum of square of residuals for each value of a_vals

for i, a in enumerate(a_vals):

    rss[i] = np.sum((fertility - a*illiteracy - b)**2)



# Plot the RSS

plt.plot(a_vals, rss, '-')

plt.xlabel('slope (children per woman / percent illiterate)')

plt.ylabel('sum of square of residuals')



plt.show()

rainfall=np.asarray([ 875.5,  648.2,  788.1,  940.3,  491.1,  743.5,  730.1,  686.5,

        878.8,  865.6,  654.9,  831.5,  798.1,  681.8,  743.8,  689.1,

        752.1,  837.2,  710.6,  749.2,  967.1,  701.2,  619. ,  747.6,

        803.4,  645.6,  804.1,  787.4,  646.8,  997.1,  774. ,  734.5,

        835. ,  840.7,  659.6,  828.3,  909.7,  856.9,  578.3,  904.2,

        883.9,  740.1,  773.9,  741.4,  866.8,  871.1,  712.5,  919.2,

        927.9,  809.4,  633.8,  626.8,  871.3,  774.3,  898.8,  789.6,

        936.3,  765.4,  882.1,  681.1,  661.3,  847.9,  683.9,  985.7,

        771.1,  736.6,  713.2,  774.5,  937.7,  694.5,  598.2,  983.8,

        700.2,  901.3,  733.5,  964.4,  609.3, 1035.2,  718. ,  688.6,

        736.8,  643.3, 1038.5,  969. ,  802.7,  876.6,  944.7,  786.6,

        770.4,  808.6,  761.3,  774.2,  559.3,  674.2,  883.6,  823.9,

        960.4,  877.8,  940.6,  831.8,  906.2,  866.5,  674.1,  998.1,

        789.3,  915. ,  737.1,  763. ,  666.7,  824.5,  913.8,  905.1,

        667.8,  747.4,  784.7,  925.4,  880.2, 1086.9,  764.4, 1050.1,

        595.2,  855.2,  726.9,  785.2,  948.8,  970.6,  896. ,  618.4,

        572.4, 1146.4,  728.2,  864.2,  793. ])
for _ in range(50):

    # Generate bootstrap sample: bs_sample

    bs_sample = np.random.choice(rainfall, size=len(rainfall))



    # Compute and plot ECDF from bootstrap sample

    x, y = ecdf(bs_sample)

    _ = plt.plot(x, y, marker='.', linestyle='none',

                 color='gray', alpha=0.1)



# Compute and plot ECDF from original data

x, y = ecdf(rainfall)

_ = plt.plot(x, y, marker='.')



# Make margins and label axes

plt.margins(0.02)

_ = plt.xlabel('yearly rainfall (mm)')

_ = plt.ylabel('ECDF')



# Show the plot

plt.show()
def bootstrap_replicate_1d(data, func):

    return func(np.random.choice(data, size=len(data)))



def draw_bs_reps(data, func, size=1):

    """Draw bootstrap replicates."""



    # Initialize array of replicates: bs_replicates

    bs_replicates = np.empty(size)



    # Generate replicates

    for i in range(size):

        bs_replicates[i] = bootstrap_replicate_1d(data,func)



    return bs_replicates
'''

Notice that the SEM we got from the known expression and the bootstrap replicates is the same and the distribution of the 

bootstrap replicates of the mean is Normal.

'''



# Take 10,000 bootstrap replicates of the mean: bs_replicates

bs_replicates = draw_bs_reps(rainfall,func=np.mean,size=10000)



# Compute and print SEM

sem = np.std(rainfall) / np.sqrt(len(rainfall))

print(sem)



# Compute and print standard deviation of bootstrap replicates

bs_std = np.std(bs_replicates)

print(bs_std)



# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, density=True)

_ = plt.xlabel('mean annual rainfall (mm)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
'''

This is not normally distributed, as it has a longer tail to the right.

Note that you can also compute a confidence interval on the variance, or any other statistic, 

using np.percentile() with your bootstrap replicates.

'''

# Generate 10,000 bootstrap replicates of the variance: bs_replicates

bs_replicates = draw_bs_reps(rainfall,np.var,size=10000)



# Put the variance in units of square centimeters

bs_replicates=bs_replicates/100



# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel('variance of annual rainfall (sq. cm)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
'''

This gives you an estimate of what the typical time between no-hitters is. It could be anywhere between 660 and 870 games.

'''

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates

bs_replicates = draw_bs_reps(nohitter_times,np.mean,10000)



# Compute the 95% confidence interval: conf_int

conf_int = np.percentile(bs_replicates,[2.5,97.5])



# Print the confidence interval

print('95% confidence interval =', conf_int, 'games')



# Plot the histogram of the replicates

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel(r'$\tau$ (games)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()

gapminder=pd.read_csv('../input/gapminder.csv')

gapminder.info()
gapminder.loc[0:200:20]
# Plotting a graph for the year 1965 for babies_per_woman vs age5_surviving(% of babies surviving till 5 years of age)



#gapminder[gapminder['year']==1965].plot(x='babies_per_woman',y='age5_surviving',kind='scatter')



# or a neat way is :



gapminder[gapminder.year==1965].plot.scatter('babies_per_woman','age5_surviving')
gapminder.iloc[1063,:]
# Define a function to show it for other years also , not just 1965



def plotyear(year):

    data=gapminder[gapminder.year==year]

    

    # show points according to population , bigger points for larger population

    # doing just 'size_param=data.population' creates big circles which cover the whole plot region , so I reduced the size of the 

    # circles exponentially for each points , hence the relative size will be maintained , after few tries 4e-6 seems appropirate

    size_param=4e-6*data.population

    

    # Colours countries by continent , done by creating a disctionary region:colour

    colour_param=data.region.map({'Africa':'skyblue' , 'Asia':'coral' , 'Europe':'palegreen' , 'America':'gold'})

    

    data.plot.scatter('babies_per_woman','age5_surviving',

                      s=size_param,

                      c=colour_param,

                      edgecolors='k',

                      linewidths=1,

                      figsize=(12,9))

    

    # plotting details

    plt.axis(ymin=50,ymax=105,xmin=0,xmax=8)

    plt.xlabel('babie sper vwoman')

    plt.ylabel('% childern alive at 5')

    

interact(plotyear,year=widgets.IntSlider(min=1950,max=2015,value=1965,step=1))

    
china1965 = pd.read_csv('../input/income-1965-china.csv')

china2015 = pd.read_csv('../input/income-2015-china.csv')

usa1965 = pd.read_csv('../input/income-1965-usa.csv')

usa2015 = pd.read_csv('../input/income-2015-usa.csv')


china1965.quantile([0.25,0.75])
china1965.quantile(0.5)

# same as median
china1965.median()
scipy.stats.percentileofscore(china1965.income,1.5)
china1965.describe()
usa1965.describe()
china1965.income.plot(kind='box')
pd.DataFrame({'usa': usa1965.log10_income, 'china': china1965.log10_income}).boxplot()
# Both works similar

#china1965.income.plot(kind='hist',histtype='step',bins=30)

china1965.income.plot.hist(bins=30,histtype='step')

plt.axvline(china1965.income.mean(),c='C1')

plt.axvline(china1965.income.median(),c='C1',linestyle='--')

plt.axvline(china1965.income.quantile(0.25),c='C1',linestyle=':')

plt.axvline(china1965.income.quantile(0.75),c='C1',linestyle=':')
china1965.income.plot(kind='hist',histtype='step',bins=30,density=True)

china1965.income.plot.density(bw_method=0.5)



plt.axis(xmin=0,xmax=3)
china1965.log10_income.plot.hist(histtype='step',bins=20)

usa1965.log10_income.plot.hist(histtype='step',bins=20)



levels = [0.25,0.5,1,2,4,8,16,32,64]

plt.xticks(np.log10(levels),levels);
china2015.log10_income.plot.hist(histtype='step',bins=20)

usa2015.log10_income.plot.hist(histtype='step',bins=20)



levels = [0.25,0.5,1,2,4,8,16,32,64]

plt.xticks(np.log10(levels),levels);
china_pop2015 = float(gapminder.query('country == "China" and year == 2015').population)

usa_pop2015 = float(gapminder.query('country == "United States" and year == 2015').population)

china2015['weight'] = china_pop2015 / len(china2015)

usa2015['weight'] = usa_pop2015 / len(usa2015)
china2015.log10_income.plot.hist(histtype='step',bins=20,weights=china2015.weight)

usa2015.log10_income.plot.hist(histtype='step',bins=20,weights=usa2015.weight)



levels = [0.25,0.5,1,2,4,8,16,32,64]

plt.xticks(np.log10(levels),levels);
italy=gapminder.query("country=='Italy'")

italy.head()
italy.plot.scatter('year','population')
gapminder.query("country=='India'").plot.scatter('year','population')
italy.plot.scatter("year", "gdp_per_day")
italy.plot.scatter("year", "gdp_per_day", logy=True)
italy.plot.scatter("gdp_per_day", "life_expectancy")
italy.plot.scatter("gdp_per_day", "life_expectancy", logx=True)

# We can see better cluttering around bottom left , when seen on logx scale
# increasing size for the years which are decade

size=np.where(italy.year % 10 ==0 , 30 ,2)



italy.plot.scatter("gdp_per_day", "life_expectancy", logx=True, s=size)

# Plotting life expectancy over the years for two countries on a single plot



data=gapminder.query("(country=='Italy') or (country=='United States')")

colour_param=np.where(data.country=='Italy','blue','green')

size=np.where(data.year % 10 ==0 , 30 ,2)



data.plot.scatter('gdp_per_day','life_expectancy',c=colour_param,logx=True,s=size)
def plotyear(nyear):

    data=gapminder.query("year== @nyear")

    #data=gapminder[gapminder['year']== nyear]

    data.plot.scatter('gdp_per_day','life_expectancy')

    

plotyear(1965)
# Repeated to highlight the difference in usage of 

# gapminder.query("year== @nyear") vs gapminder[gapminder['year']== nyear]

def plotyear(nyear):

    #data=gapminder.query("year== nyear")

    data=gapminder[gapminder['year']== nyear]

    data.plot.scatter('gdp_per_day','life_expectancy')

    

plotyear(1965)
def plotyear(nyear):

    #data=gapminder.query("year== nyear")

    data=gapminder[gapminder['year']== nyear]

    data.plot.scatter('gdp_per_day','life_expectancy',logx=True)

    

plotyear(1965)
# Making the above visually informative

# visualizing 4 variables together : population ,age5_surviving , gdp_per_day , life_expectancy

def plotyear(nyear):

    data=gapminder[gapminder['year']== nyear]

    size_param=4e-6*data.population

    colour_param=data.age5_surviving

    data.plot.scatter('gdp_per_day','life_expectancy',logx=True,s=size_param,

                      c=colour_param,

                      colormap=matplotlib.cm.get_cmap('Blues_r'),

                      edgecolors='k',

                      linewidths=1,sharex=False

                      )

    

plotyear(1965)
# Making the above visually informative

# visualizing 5 variables together : population ,age5_surviving , gdp_per_day , life_expectancy , region

def plotyear(year):

    data=gapminder[gapminder['year']== year]

    size_param=4e-6*data.population

    colour_param=data.age5_surviving

    edgecolor = data.region.map({'Africa': 'skyblue','Europe': 'gold','America': 'palegreen','Asia': 'coral'})

    data.plot.scatter('gdp_per_day','life_expectancy',logx=True,s=size_param,

                      c=colour_param,

                      colormap=matplotlib.cm.get_cmap('Blues_r'),

                      edgecolors=edgecolor,

                      linewidths=1,sharex=False,

                      figsize=(10,6)

                      )

    

plotyear(1965)
interact(plotyear,year=range(1965,2016,10))
def plotyear(year):

    data=gapminder[gapminder['year']== year]

    size_param=4e-6*data.population

    colour_param=data.age5_surviving

    edgecolor = data.region.map({'Africa': 'skyblue','Europe': 'gold','America': 'palegreen','Asia': 'coral'})

    data.plot.scatter('gdp_per_day','life_expectancy',logx=True,s=size_param,

                      c=colour_param,

                      colormap=matplotlib.cm.get_cmap('Blues_r'),

                      edgecolors=edgecolor,

                      linewidths=1,sharex=False,

                      figsize=(10,6)

                      )

    for level in [4,16,64]:

        plt.axvline(level,linestyle=':',color='k')

    

plotyear(1965)
# gapminder['log10_gdp_per_day'] = np.log10(data['gdp_per_day'])

# data = gapminder.loc[gapminder.year == 2015,['log10_gdp_per_day','life_expectancy','age5_surviving','babies_per_woman']]





# #scatter_matrix(data,figsize=(9,9))

# #pd.plotting.scatter_matrix(data)

# data
smoking = pd.read_csv('../input/whickham.csv')

smoking.head()
pd.DataFrame(smoking.smoker.value_counts())
pd.DataFrame(smoking.outcome.value_counts())
pd.DataFrame(smoking.outcome.value_counts(normalize=True))
bysmoker=smoking.groupby('smoker').outcome.value_counts()

bysmoker
bysmoker.index
bysmoker.unstack()
smoking['AgeGroup']=pd.cut(smoking.age,[0,30,40,53,64],labels=['0-30','30-40','40-53','53-64'])

smoking['AgeGroup'].head()
byage=smoking.groupby(['AgeGroup','smoker']).outcome.value_counts(normalize=True)

byage.head()
byage.unstack().drop('Dead',axis=1)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

smoking.outcome.value_counts().plot.pie()

plt.title('outcome')



plt.subplot(1,2,2)

smoking.smoker.value_counts().plot.pie()

plt.title('smoker')
bysmoker.plot.bar()
bysmoker.unstack().plot.bar()
bysmoker.unstack().plot.bar(stacked=True)
byage.unstack().plot.bar(stacked=True)
byage.unstack().drop('Dead',axis=1).unstack().plot.bar()

# Make it further neat
byage2=byage.unstack().drop('Dead',axis=1).unstack()

byage2
byage2.columns
byage2.columns=['No','Yes']

byage2.columns.name='smoker'
byage2
byage2.plot.bar()
poll = pd.read_csv('../input/poll.csv')

poll.head()
poll.vote.value_counts(normalize=True)
np.random.rand(5)
np.random.rand(5) < 0.51
np.where(np.random.rand(5) < 0.51 , 'Brown','Green')
def sample(brown,n=1000):

    return pd.DataFrame({'vote': np.where(np.random.rand(n) < brown , 'Brown','Green')})



s=sample(0.51)
s.vote.value_counts(normalize=True)
dist=pd.DataFrame([sample(0.51).vote.value_counts(normalize=True) for i in range(1000)])

dist.head()
dist.Brown.plot.hist(histtype='step',bins=20)
# Revisit

def samplingdist(brown,n=1000):

    return pd.DataFrame([sample(brown,n).vote.value_counts(normalize=True) for i in range(1000)])
def quantiles(brown,n=1000):

    dist = samplingdist(brown,n)

    return dist.Brown.quantile(0.025), dist.Brown.quantile(0.975)
quantiles(0.50)
quantiles(0.48)
dist = samplingdist(0.50,10000)
dist.Brown.hist(histtype='step')
largepoll = pd.read_csv('../input/poll-larger.csv')

largepoll.vote.value_counts(normalize=True)
pop = pd.read_csv('../input/grades.csv')
pop.head()
pop.grade.plot.hist(histtype='step')

pop.describe()

# The histogram does not have a very well defined structure . It has a mean of 5.5 though.
'''

What can we say about the true mean value.

This time we cannot build a confidence interval by simulating the sampling distribution because we do not know how to describe it. 

And, indeed, given the observed histogram it is unlikely that it has a simple form such as a normal distribution.



What we'll do is to estimate the uncertainty of our statistic, the mean, by generating a large family of samples from the one we have. 

And then, characterizing the distribution of the mean over this family.



Each sample in the family is prepared as follow: we draw grades randomly for our single existing sample allowing the same grade to be drawn more than once.

Technically speaking, we are sampling with replacement

'''
# We took 100 samples from the given 100 points , but with replacement

pop.sample(100,replace=True).describe()



# The mean for this bootstrapped sample is a bit different from 5.508561 (mean of 100 samples originally)
'''

We repeat the bootstrap sampling for 1000 times and record the means in a dataframe for analysis

'''

bootstrap=pd.DataFrame({'meangrade':[pop.sample(100,replace=True).grade.mean() for i in range(1000)]})
bootstrap.meangrade.plot.hist(histtype='step')

plt.axvline(bootstrap.meangrade.mean(),color='yellow',linestyle='-')

plt.axvline(pop.grade.mean(),color='red')

'''

Both means are same , but there is a significant spread over the bootstrapped mean

'''
bootstrap.meangrade.quantile(0.025),bootstrap.meangrade.quantile(0.975)
# position of 8 water pumps in central london

pumps = pd.read_csv('../input/pumps.csv')

pumps
# Deaths by area and the nearest pump to the area

cholera = pd.read_csv('../input/cholera.csv')

cholera.head()
# Plotting pumps and deaths

plt.figure(figsize=(6,6))

plt.scatter(cholera.x,cholera.y,s=3)

plt.scatter(pumps.x,pumps.y)
cholera.closest.value_counts()
'''

More deaths near pump zero may be due to the fact that more people might be living near it as compared to other pumps



'''

cholera.groupby('closest').deaths.sum()



'''

Let's simulate each death randomly, proportionally to the population of each area.

0: 340

Total : 489

'''

def simulate(n):

    return pd.DataFrame({'closest': np.random.choice([0,1,4,5],size=n,p=[0.65,0.15,0.10,0.10])})
simulate(489).closest.value_counts()
'''

So we get something close to what we actually observed in the true data. What we need now is the sampling distribution 

of the number of deaths in area zero.I will extract the count for area zero, repeat the operation 10,000 times, and 

enclose the result in a DataFrame.

'''

sampling = pd.DataFrame({'counts': [simulate(489).closest.value_counts()[0] for i in range(10000)]})
# Lets look at the histogram

sampling.counts.plot.hist(histtype='step')
'''

We have generated this distribution under the null hypothesis that the pumps have nothing to do with cholera, 

and the deaths occur simply proportionally to population. We can now compare this distribution with the observed 

number of 340 deaths in area zero.More precisely, we evaluate at what quantile we find 340 in this null hypothesis sampling distribution.



So 340 is a very extreme value, which we would not expect from the null scenario. In fact, we'd expect it only 1.70(100-98.3) percent 

of the time.This is known as the P value, the smaller the P value, the more strongly we can reject the null hypothesis.

'''

scipy.stats.percentileofscore(sampling.counts,340)

poll.vote.value_counts(normalize=True)
'''

In the smaller poll Brown had a seeming majority of votes, so here the null hypothesis will be that Green wins or ties the election, 

so the true Brown fraction would be 0.50 or less. We need to find out whether a Brown proportion of 0.511 is an extreme result 

if the null hypothesis holds. So we compute the sampling distribution of the proportion, and get a true Brown fraction of 0.50. 



If it's lower than that the P value will be even lower. 

'''

def sample(brown, n=1000):

    return pd.DataFrame({'vote': np.where(np.random.rand(n) < brown,'Brown','Green')})
dist = pd.DataFrame({'Brown': [sample(0.50,1000).vote.value_counts(normalize=True)['Brown'] for i in range(10000)]})
dist.Brown.hist(histtype='step',bins=20)
100 - scipy.stats.percentileofscore(dist.Brown,0.511)
largepoll.vote.value_counts(normalize=True)
dist = pd.DataFrame({'Green': [sample(0.50,10000).vote.value_counts(normalize=True)['Green'] for i in range(1000)]})
dist.Green.hist(histtype='step',bins=20)

plt.axvline(0.5181,c='C1')