# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline 
# ^^^^^^^^^^^^^^^^^ Tells notebook to print on screen in kernel and not save it in a separate file
import matplotlib.pyplot as plt # graphing data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
for i in range(10):
    print(i)
count = 100
x = 2*np.random.rand(count) - 1
plt.plot(x, 'bo')
count =  100
noise = 0.1 * np.random.rand(count) # Creates 100 random numbers between 0 and 1 that are multiplied by 0.1
#Numbers are between 0 and 0.1

#Makes variable imperfect by adding random number to it
#The random number comes from noise
noisy_x = x + noise

#Creates quadratic equation: y = 50x^2 + 2x
y = 50 * (noisy_x) * (noisy_x) + 2 * (noisy_x)

#Plots graph of x and y with red cirles
plt.plot(x, y, 'ro')
"""
CENTRAL LIMIT THEOREM (from Stats)
Play with the number of random "events" to get 
visual proof of central limit theorem. 

Lets take some random unrelated "events" that produce values 
between 0 and 100. 

Lets have 1000 trials ( each try has a set of "events" )

Find the average of the events in each trial and plot them. 

If number of event = 1, then avarage across trials is 
uniformly distributed between 0 and 100 

As you increase the number of events, the distribution of 
the average, starts converging to the center (~50)
"""
events = 5 #Number of random unrelated events
trials = 1000 #Number of tries for each set of random events above
z= np.random.randint(0,100,[events,trials]) #integers between 0 and 100, randomly chose for each variable, and each try
total = sum(z[:])/events #For each trial, Average the set of events 

#Plot a histogram with 11 bins for values of "averages" stored in array total
plt.hist(total, bins=11, normed=0)
plt.axis([0,100,0,300])
plt.xlabel('average')
plt.ylabel('frequency')
plt.show()