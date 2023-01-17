# RUN ALL THE CODE BEFORE YOU START

import numpy as np

from matplotlib.pylab import plt #load plot library

# indicate the output of plotting function is printed to the notebook

%matplotlib inline 





def create_random_walk():

    x = np.random.choice([-1,1],size=100, replace=True) # Sample with replacement from (-1, 1)

    return np.cumsum(x) # Return the cumulative sum of the elements

X = create_random_walk()

Y = create_random_walk()

Z = create_random_walk()



# Plotting functionality starts here

plt.plot(X)

plt.plot(Y)

plt.plot(Z)

plt.show()
plt.plot(X, '-.', color="#333333")

plt.plot(Y, '-.', color="chocolate")

plt.plot(Z, '-.', color="green")

plt.show()
plt.plot(X, label="X")

plt.plot(Y, label="Y")

plt.plot(Z, label="Z")

# Add legend

plt.legend(loc='lower left')

# Add title and x, y labels

plt.title("Random Walk Example", fontsize=16, fontweight='bold')

plt.suptitle("Random Walk Suptitle", fontsize=10)

plt.xlabel("Number of Steps")

plt.ylabel("Accumulative Sum")

plt.show()
# RUN ALL THE CODE BEFORE YOU START

import numpy as np

from matplotlib.pylab import plt #load plot library

# indicate the output of plotting function is printed to the notebook

%matplotlib inline 



x = np.linspace(0, 6, 100)

y_1 = 5*x

y_2 = np.power(x, 2)

y_3 = np.exp(x/1.5)

plt.plot(x, y_1)

plt.plot(x, y_2)

plt.plot(x, y_3)

plt.show()
plt.plot(x, y_1, '-.', color="#333333")

plt.plot(x, y_2, '--', color="#999999")

plt.plot(x, y_3, '-', color="#aaaaaa")

plt.show()
plt.plot(x, y_1, '-.', color="#333333", label="$y=5 * x$")

plt.plot(x, y_2, '--', color="#999999", label="$y=x^2$")

plt.plot(x, y_3, '-', color="#aaaaaa", label="$y=e^{3/2x}$") # TODO: Fix the notation here

plt.legend(loc="upper left")



plt.title("Several Functions", fontsize=16, fontweight='bold')

plt.xlabel("X Values")

plt.ylabel("Y Values")



plt.show()