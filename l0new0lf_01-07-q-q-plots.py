import numpy as np 

import pylab 

import scipy.stats as stats



import matplotlib.pyplot as plt

import seaborn as sns
# gen data1 and data2 --- loc ~ Mean and scale ~ std-dev

# note: params are different

std_normal_data_1k = np.random.normal(loc = 0, scale = 1, size=1000)

normal_data_3k = np.random.normal(loc = 20, scale = 100, size=3000)



# generate percentiles

percentiles_std_norm_data = [np.percentile(std_normal_data_1k,i) for i in range(0,101)]

percentiles_norm_data = [np.percentile(normal_data_3k,i) for i in range(0,101)]
# plot percentiles

plt.scatter(percentiles_std_norm_data, percentiles_norm_data)



plt.xlabel("std normal data (1k samples)")

plt.ylabel("normal data (3k samples)")

plt.show()
# method 2: using stats library

# check if raw data `normal_data_3k` is `norm` distribures

stats.probplot(normal_data_3k, dist="norm", plot=pylab)



pylab.show()
normal_data_1000 = np.random.normal(loc = 20, scale = 100, size=1000)

normal_data_100  = np.random.normal(loc = 20, scale = 100, size=100)

normal_data_50   = np.random.normal(loc = 20, scale = 100, size=10)



plt.close()

stats.probplot(normal_data_1000, dist="norm", plot=pylab)

pylab.show()



plt.close()

stats.probplot(normal_data_100, dist="norm", plot=pylab)

pylab.show()



plt.close()

stats.probplot(normal_data_50, dist="norm", plot=pylab)

pylab.show()
measurements = np.random.uniform(low=-1, high=1, size=100) 



# uniform distribution vs. normal distribution

stats.probplot(measurements, dist="norm", plot=pylab)

pylab.show()