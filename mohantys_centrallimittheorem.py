

# generate random dice rolls

from numpy.random import seed

from numpy.random import randint

from numpy import mean

# seed the random number generator

seed(1)

# generate a sample of die rolls

rolls = randint(1, 7, 50)

print(rolls)

print(mean(rolls))

# demonstration of the central limit theorem

from numpy.random import seed

from numpy.random import randint

from numpy import mean

from matplotlib import pyplot

# seed the random number generator

seed(1)

# calculate the mean of 50 dice rolls 1000 times

means = [mean(randint(1, 7, 50)) for _ in range(1000)]

# plot the distribution of sample means

pyplot.hist(means)

pyplot.show()