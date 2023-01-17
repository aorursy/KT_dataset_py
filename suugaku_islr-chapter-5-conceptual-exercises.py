import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
n = np.arange(1, 100001)

probs = 1 - (1 - (1/n))**n

fig, ax = plt.subplots(figsize = (10, 8))

ax.plot(n, probs)

ax.axhline(1 - 1/np.e, linestyle = ":", linewidth = 3, color = "orange", label = "1 - 1/e")

ax.legend(loc = "best")

ax.set(xscale = "log", xlabel = "Number of observations, n", 

       ylabel = "Probability jth observation is in the bootstrap sample of size n");
# Performing 10,000 simulations with n = 100 observations



# Set seed for reproducible results

np.random.seed(312)

# Create matrix where the row m, column n entry is the mth observation from the nth bootstrap sample

bootstraps = np.trunc(np.random.random_sample((100, 10000))*100 + 1)

# Count the number of the jth observation (e.g. j = 4) is in each bootstrap sample

num_j = (bootstraps == 4).sum(axis = 0)

# Compute proportion of bootstrap samples which contain the jth observation at least once

(num_j > 0).mean()
# Performing 1,000,000 simulations with n = 100 observations



# Set seed for reproducible results

np.random.seed(312)

# Create matrix where the row m, column n entry is the mth observation from the nth bootstrap sample

bootstraps = np.trunc(np.random.random_sample((100, 1000000))*100 + 1)

# Count the number of the jth observation (e.g. j = 4) is in each bootstrap sample

num_j = (bootstraps == 4).sum(axis = 0)

# Compute proportion of bootstrap samples which contain the jth observation at least once

(num_j > 0).mean()