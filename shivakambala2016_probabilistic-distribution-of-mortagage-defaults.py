# import required libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Defining a Bernoulli trail function

def Perform_Bernoulli_trail(num_events,p):
    num_success = 0
    for i in range(num_events):
        if np.random.random() < p:
            num_success = num_success + 1
    return num_success

            
num_defaults = np.empty(1000)
np.random.seed(22)
for i in range(1000):
    num_defaults[i] = Perform_Bernoulli_trail(100,0.05)
    
    
num_defaults
# plotting the results using  a histogram

plt.hist(num_defaults)
plt.xlabel("Number of defaluts out of 100 loans")
plt.ylabel("Frequency")
plt.margins(0.02)
plt.style.use('seaborn-darkgrid')

