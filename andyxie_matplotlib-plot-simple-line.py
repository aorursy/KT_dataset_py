# RUN ALL THE CODE BEFORE YOU START

import numpy as np

from matplotlib.pylab import plt #load plot library

# indicate the output of plotting function is printed to the notebook

%matplotlib inline 
x = np.linspace(-3, 3, 100)

plt.plot(x)

plt.show()
plt.plot(x, 'r-.')

plt.show()
import numpy as np

from matplotlib.pylab import plt

%matplotlib inline



X = np.linspace(-3, 3, 100) # Create an array of 100 numbers between -3 and 3

X[:10]

Y = np.power(X, 2) # Calculate Y

plt.plot(X, Y)

plt.show()
plt.plot(X, Y, 'k--')

plt.show()
X = np.linspace(-3, 3, 100)

error = np.random.normal(0,0.4,100) # Make an array of 100 numbers from normal distribution

                                    #with mean = 0, std = 0.4

Y = np.power(X, 2) + error

plt.plot(X, Y, 'r.') # r. means red dots

plt.show()