import numpy
def sigmoid(z):

    return 1.0 / (1.0 + numpy.exp(-z))
z = numpy.linspace(-10, 10)



y = sigmoid(z)
%matplotlib inline

import matplotlib.pyplot as pyplot



fig1 = pyplot.figure()

axes1 = pyplot.axes()

axes1.set_xlabel('z')

axes1.set_ylabel('y')

plot1 = axes1.plot(z, y, label='y = sigmoid(z)')

legend1 = axes1.legend()

z_small = numpy.linspace(-1, 1)

y_small = 0.25 * z_small + 0.5
axes1.plot(z_small, 0.25 * z_small + 0.5, label='y = 0.25z + 0.5')

axes1.legend()

fig1
y_step = sigmoid(100 * z)

axes1.plot(z, y_step, label='y = sigmoid(100 * z)')

axes1.legend()

fig1