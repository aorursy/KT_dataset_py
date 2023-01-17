# generate and plot an idealized gaussian

from numpy import arange

from matplotlib import pyplot

from scipy.stats import norm

# x-axis for the plot

x_axis = arange(-3, 3, 0.001)

# y-axis as the gaussian

y_axis = norm.pdf(x_axis, 0, 1)

# plot data

pyplot.plot(x_axis, y_axis)

pyplot.show()
# generate a sample of random gaussians

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# histogram of generated data

pyplot.hist(data)

pyplot.show()
#generate a sample of random gaussians

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# histogram of generated data

pyplot.hist(data, bins=100)

pyplot.show()
# calculate the mean of a sample

from numpy.random import seed

from numpy.random import randn

from numpy import mean

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# calculate mean

result = mean(data)

print('Mean: %.3f' % result)
# calculate the median of a sample

from numpy.random import seed

from numpy.random import randn

from numpy import median

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# calculate median

result = median(data)

print('Median: %.3f' % result)
# generate and plot gaussians with different variance

from numpy import arange

from matplotlib import pyplot

from scipy.stats import norm

# x-axis for the plot

x_axis = arange(-3, 3, 0.001)

# plot low variance

pyplot.plot(x_axis, norm.pdf(x_axis, 0, 0.5))

# plot high variance

pyplot.plot(x_axis, norm.pdf(x_axis, 0, 1))

pyplot.show()
# calculate the variance of a sample

from numpy.random import seed

from numpy.random import randn

from numpy import var

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# calculate variance

result = var(data)

print('Variance: %.3f' % result)
# calculate the standard deviation of a sample

from numpy.random import seed

from numpy.random import randn

from numpy import std

# seed the random number generator

seed(1)

# generate univariate observations

data = 5 * randn(10000) + 50

# calculate standard deviation

result = std(data)

print('Standard Deviation: %.3f' % result)