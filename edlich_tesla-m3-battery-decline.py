# Analyse the TESLA Model 3 Standard Battery decline with data from a friend 



from scipy import stats

from numpy import genfromtxt

import matplotlib.pyplot as plt



# import CSV

my_data = genfromtxt('../input/TESLA-M3-BATT-DECL.csv', delimiter=',')



stats.describe(my_data)
plt.plot(my_data)

plt.show()