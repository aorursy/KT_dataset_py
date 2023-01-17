import pandas as pd

from scipy.stats import ttest_ind

from scipy.stats import probplot

import pylab

import matplotlib.pyplot as plt

# Read in the cereal data

cereal = pd.read_csv('../input/cereal.csv')

# Print the first few lines

cereal.head()
probplot(cereal['calories'], dist='norm', plot=pylab)
probplot(cereal['potass'], dist='norm', plot=pylab)
# Get the hot cereals

cereal_hot = cereal['potass'][cereal['type'] == 'H']

# Get the cold cereals

cereal_cold = cereal['potass'][cereal['type'] == 'C']
ttest_ind(cereal_hot, cereal_cold, equal_var=False)
print('Hot cereal average:')

print(cereal_hot.mean())

print('Cold cereal average:')

print(cereal_cold.mean())
import seaborn as sns
# Plotting both hot and cold cereals together

sns.distplot(cereal_hot, label='Hot Cereals')

sns.distplot(cereal_cold, label='Cold Cereals')

plt.legend()
# Plotting the hot cereals

sns.distplot(cereal_hot, color='r', label='Hot Cereals')

plt.legend()
# Plot the cold cereals

sns.distplot(cereal_cold, color='b', label='Cold Cereals')

plt.legend()