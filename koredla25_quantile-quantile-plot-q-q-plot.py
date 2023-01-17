import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
# A standard Normal distribution, which means mean =0 and standard deviation =1
df=np.random.normal(loc=0,scale=1,size=1000)
df.size
sns.distplot(df)
# 0 to 100th percentiles of std-norm
for i in range(0,101):
    print(i, np.percentile(df,i))
# generate 100 samples from N(20,5)
measurements = np.random.normal(loc=20,scale=5,size=50000)
measurements.size
sns.distplot(measurements)
# Plot the both distributions 
import pylab
stats.probplot(measurements,dist='norm',plot=pylab)
pylab.show
# Now we will cross check with Uniform Distribution

# generate 100 sanples from N(20,5)

# Uniform Distribution 

measurements = np.random.uniform(low=-1, high=1, size=10000) 
sns.distplot(measurements)
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
rvs = stats.norm(loc=0,scale=1)
# We can plot exponential distribution also
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.distplot(stats.expon().rvs(size=10000))
# Let's compare them
expon_rvs = stats.expon().rvs(size=100000)
normal_rvs = stats.norm().rvs(size=100000)
stats.probplot(x=expon_rvs, dist=stats.norm(), plot=plt)
# I will tell you later what Pareto distribution is.
pareto_rvs = stats.pareto(b=2.62).rvs(size=1000000)
stats.probplot(x=pareto_rvs, dist=stats.expon(), plot=plt)
plt.show()
