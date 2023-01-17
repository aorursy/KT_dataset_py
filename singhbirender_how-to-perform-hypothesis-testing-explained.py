import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import scipy.stats as stats

import math

%matplotlib inline
cust = pd.read_csv("../input/cust_seg.csv")

cust.head()
cust.columns
cust.info()
cust.describe()
cust.Latest_mon_usage.mean()
cust.Latest_mon_usage.std()
stats.ttest_1samp(a = cust.Latest_mon_usage, popmean = 50) # pop mean is the hypothetical value
cust.Latest_mon_usage.mean()
print(cust.pre_usage.mean())

print(cust.Post_usage_1month.mean())

print(cust.post_usage_2ndmonth.mean())
stats.ttest_rel(a = cust.pre_usage, b = cust.Post_usage_1month)
Males_spend = cust.Post_usage_1month[cust.sex == 0]

Females_spend = cust.Post_usage_1month[cust.sex == 1]
print(Males_spend.head())

print(Females_spend.head())
print(Males_spend.mean())

print(Females_spend.mean())
print(Males_spend.std())

print(Females_spend.std())
stats.ttest_ind(a = Males_spend, b = Females_spend, equal_var = False)

# equal_var Assume samples have equal variance?
# we can use ANOVA as well.

stats.f_oneway(Males_spend, Females_spend)
cust.segment.value_counts()
s1 = cust.Latest_mon_usage[cust.segment == 1]

s2 = cust.Latest_mon_usage[cust.segment == 2]

s3 = cust.Latest_mon_usage[cust.segment == 3]



# perform ANOVA test

stats.f_oneway(s1, s2, s3)
print(s1.mean(), s2.mean(), s3.mean() )
t = pd.crosstab(cust.segment, cust.region, margins = True)

t

# actual distribution between segment and region
stats.chi2_contingency(observed = t)
print(np.corrcoef(cust.Latest_mon_usage, cust.Post_usage_1month))
print(stats.stats.pearsonr(cust.Latest_mon_usage, cust.Post_usage_1month))