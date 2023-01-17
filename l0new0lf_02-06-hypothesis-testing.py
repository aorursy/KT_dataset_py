from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
# 1. one sample t-test

# =====================



# population:

mu, sigma = 100, 30 # mean and std-dev

population = np.random.normal(mu, sigma, 1000)

# Note:

# Population mean will be known before hyp-test.

# Infact, we will be checking if  `mu = s_bar`



# sample:

sample_norm = np.random.choice(population, size=30)



# check if sample is normally distributed

# can use q-q plot. (Avoiding it cz, as 

# it is randomly sample from uniform distribution to

# >>sufficient size<<, it will be normally distributed too)

# In real data, check it cz, we may not know population

# (Must follow assumptions for t-test)

sns.distplot(sample_norm)

plt.title("sample")

plt.show()
# Define,

# H_o: s_bar = mu

# H_a: s_bar != mu

# level_of_significance = 5% = 0.05



# one sample t-test -- generates p-val

stats.ttest_1samp(sample_norm, mu) # note: mu (population mean) -- checks if `s_bar == mu`
# Let us check with another sample w/ 

# mean little shifted from population mean.

# As sample_mean is shifted from population

# _mean, it should return lesser p-val

# ----------------------------------------



# sample 2:

sample_norm_2 = np.random.normal(mu+10, sigma, 30)



# it is normally distributed.

# No need to check



# H_o: s_bar = mu

# H_a: s_bar != mu

# level_of_significance = 5% = 0.05



# one sample t-test -- generates p-val

stats.ttest_1samp(sample_norm_2, mu)
# 2. two sample independant t-test

# ================================



# population:

mu, sigma = 100, 30 # mean and std-dev

population = np.random.normal(mu, sigma, 1000)

# Note:

# Population mean will not be known before hyp-test.

# only norm_sample1 and norm_sample2 will be known

# We will be checking if  `s_bar_1 = s_bar_2`





# sample1 and sample 2:

norm_sample_1 = np.random.choice(population, size=70)

norm_sample_2 = np.random.choice(population, size=70)



# Note: as randomly sampled from normal distribution

# to >>sufficient size<<, they are normal as well. If real

# world data, have to check if normal or not using

# q-q plot or any other method.

# Here, as we already know, not checking
# Define,

# H_o: s_bar_1 = s_bar_2

# H_a: s_bar_1 != s_bar_2

# level_of_significance = 5% = 0.05



# two sample independant t-test

stats.ttest_ind(norm_sample_1, norm_sample_2)
# 2. two sample paired/relational t-test

# ======================================



# sample_1 and sample_2

sample_1_before_pill = [68,45,46,34,23,67,80,120,34,54,68] 

sample_2_after_pill  = [28,25,26,24,13,37,30,30,54,34,38]
# assumption: must be guassian norm.

# check using q-q plot or k-s test

fig, axarr = plt.subplots(1, 2)

fig.set_size_inches(15,4)



stats.probplot(sample_1_before_pill, dist="norm", plot=axarr[0])

axarr[0].title.set_text("sample_1_before_pill")



stats.probplot(sample_2_after_pill, dist="norm", plot=axarr[1])

axarr[1].title.set_text(("sample_2_after_pill"))



plt.show()
# Define,

# H_o: s_bar_1 = s_bar_2

# H_a: s_bar_1 != s_bar_2

# level_of_significance = 5% = 0.05



# two sample paired/relational t-test

stats.ttest_rel(sample_1_before_pill, sample_2_after_pill)