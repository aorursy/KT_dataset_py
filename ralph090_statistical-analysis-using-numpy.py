import numpy as np

''' Generating a random dataset which have been assigned to a memory location 

with an identifier as employment

Centered around 40,000 having a normal distribution and

Standard Deviation of 10,000, 

with 20,000 data points'''



employment = np.random.normal(40000, 10000, 20000)





%matplotlib inline

import matplotlib.pyplot as plt

plt.title('Normal Distribution')

plt.hist(employment, 60,density=True, facecolor='black', alpha=0.9)

plt.show()
np.mean(employment)
np.median(employment)
n_employment = np.append(employment, [99999999])

%matplotlib inline

import matplotlib.pyplot as plt

plt.title('Normal distribution with an Outlier')

plt.hist(n_employment, 100,density=True, facecolor='black', alpha=0.9)



plt.show()


print(r"Before adding an outlier Mean: {}".format(np.mean(employment)))

print(r"After adding an outlier Mean: {}".format(np.mean(n_employment)))



print("___________________________________________________________________")



print(r"Before adding an outlier Median: {}".format(np.median(employment)))

print(r"After adding an outlier Median: {}".format(np.median(n_employment)))

persons = np.random.randint(10, high=100, size=500) 

#Lowest (signed) integer=10 & Highest (signed) integer =100

persons

from scipy import stats

stats.mode(persons) # To Calculate Mode