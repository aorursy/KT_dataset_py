import matplotlib.pyplot as plt
import numpy as np
import statistics

group = np.array(['R','Bl','Y','G','Br','O','P','V'])
colour = np.array(['r','b','yellow','g','brown','orange', 'pink','violet'])
sizegrp = np.array([75, 77, 77, 85, 91, 88, 69, 73])

total = 635
freqprob = np.divide(sizegrp,total) #frequentist probability
stdgrp = statistics.stdev(sizegrp)/total #sample standard deviation
n = len(group) # number of different colours

xi = np.array([0,1,2,3,4,5,6,7])
yi = freqprob

def mean(v):
    return sum(v)/n

def s(v1,v2):
    return sum(v1 * v2) - n * mean(v1) * mean(v2)

ahat = s(xi,yi)/s(xi,xi)
bhat = mean(yi) - mean(xi) * ahat
y = ahat * xi + bhat
probmean = np.array([mean(yi)]*8)

rho = s(xi,yi)/np.sqrt(s(xi,xi) * s(yi,yi))

plt.subplot(1,2,1)
plt.plot(xi, y, label = 'Regression line', color = 'cyan')
plt.plot(xi, probmean, label = 'Mean', color = 'black')
plt.fill_between(xi, probmean - stdgrp, probmean + stdgrp, color = 'lightgray')
plt.scatter(group, yi, color = colour)
plt.grid()
plt.legend()
plt.ylabel("Probabilty")
plt.xlabel("Colour of Smarties")

plt.subplot(1,2,2)
plt.plot(xi, probmean, label = 'Mean', color = 'black', linestyle = '-')
plt.fill_between(xi, probmean - stdgrp, probmean + stdgrp, color = 'lightgray')
plt.scatter(group, yi, color = colour)
plt.ylim([0,1])

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
