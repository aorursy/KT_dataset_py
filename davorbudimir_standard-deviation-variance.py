# Standard deviation is a number that describes how spread out the values are.
# Variance is another number that indicates how spread out the values are.

import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.std(speed)

print(x)

x = numpy.var(speed)

print(x)
