# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # advanced data visualization

sns.set() # Using sns graphing style

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
# Reading data

# Compiled data
data_1975 = pd.read_csv('../input/beaks-1975/finch_beaks_1975.csv')
data_2012 = pd.read_csv('../input/beaks-2012/finch_beaks_2012.csv')

# Summarized data
df = pd.read_csv('../input/dryad-darwins-finches/Data for Dryad.txt', sep='\t')
# Checking df.head from summarized data
df.head()
# Preparing working vectors on subsets selecting scandens species from 1975 and 2012

data_2012.columns = data_1975.columns
data_2012['year'] = 2012
data_1975['year'] = 1975
concat_df = pd.concat([data_1975, data_2012])

# Generating subsets
sc_1975 = concat_df[(concat_df['species'] == 'scandens') & (concat_df['year'] == 1975)]
sc_2012 = concat_df[(concat_df['species'] == 'scandens') & (concat_df['year'] == 2012)]
# Plotting scandens distributions

concat_sc = concat_df[concat_df['species'] == 'scandens']
_ = plt.figure(figsize=(8,6))
_ = sns.swarmplot(x='year', y='Beak depth, mm', data=concat_sc, size=4);
_ = plt.ylabel('Beak depth (mm)')
_ = plt.xlabel('Year')
plt.tight_layout()
# Plotting fortis distributions
concat_sc = concat_df[concat_df['species'] == 'fortis']
_ = plt.figure(figsize=(8,6))
_ = sns.swarmplot(x='year', y='Beak depth, mm', data=concat_sc, size=4);
_ = plt.ylabel('Beak depth (mm)')
_ = plt.xlabel('Year')
plt.tight_layout()
# Defining our functions from Data Camp's power toolbox

# ECDF stands for empirical cumulative distribution function.  
def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.
    
    It assigns a probability of to each datum (x axis), orders the data from smallest to largest in value, 
    and calculates the sum of the assigned probabilities up to and including each datum (x axis).
    """
    
    # Number of data points: n
    n = len(data)
    
    # x-data for the ECDF: x
    x = np.sort(data)
    
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    
    return x, y

def bootstrap_replicate_1d(data, func):
    """
    Compute and return a bootstrap replicate, which is a statistical value according to parameter 'func'
    on a randomized numpy array based on the given 'data'
    """
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """
    Draw 'size' numbers of bootstrap replicates.
    """
    
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
# Plotting an ECDF for years 1975 and 2012
x_1975, y_1975 = ecdf(data_1975['Beak depth, mm'])
x_2012, y_2012 = ecdf(data_2012['Beak depth, mm'])
_ = plt.figure(figsize=(10,8))
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margings
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('Beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('2012', '1975'), loc='lower right')
# Making aliases
bd_1975 = np.array(sc_1975['Beak depth, mm'])
bd_2012 = np.array(sc_2012['Beak depth, mm'])

# Computing confidence intervals

"""if we repeated the measurements over and over again, 
95% of the observed values would lie withing the 95% confidence interval"""

# Compute the observed difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')
# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# This is done because we are assuming in our Ho that means are equal!
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value: p
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p-value = {0:.4f}'.format(p))
_ = plt.figure(figsize=(10,8))
_ = plt.scatter(x=sc_1975['Beak length, mm'], y=sc_1975['Beak depth, mm']);
_ = plt.xlabel('Beak length (mm)')
_ = plt.ylabel('Beak depth (mm)')

# Compute observed correlation: obs_corr_1975
obs_corr_1975 = sc_1975[['Beak length, mm', 'Beak depth, mm']].corr().iloc[0, 1]
print("Pearson correlation =", obs_corr_1975)
# The bootstrap test will be done by permuting Beak depth attribute while keeping Beak lenght the same, good practice for correlation bootstrap
# Note that the .iloc[] is used to extract the correlation value ignoring the identity values from the correlation matrix

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    beak_depth_permuted = np.random.permutation(sc_1975['Beak depth, mm'].as_matrix())

    # Compute Pearson correlation
    # Note that here np.corrcoef is used since we're working with arrays instead with a Data Frame
    # Therefore we use [0, 1] to select the correct correlation value from the correlation matrix
    perm_replicates[i] = np.corrcoef(beak_depth_permuted, sc_1975['Beak length, mm'].as_matrix())[1, 0]

# Compute p-value: p
p = np.sum(perm_replicates >= obs_corr_1975)/len(perm_replicates)
print('p-val =', p)