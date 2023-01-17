"""
Percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than.
"""

import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.percentile(speed, 0)
print(x)

x = numpy.percentile(speed, 10)
print(x)

x = numpy.percentile(speed, 25)
print(x)

x = numpy.percentile(speed, 50)
print(x)

x = numpy.percentile(speed, 75)
print(x)

x = numpy.percentile(speed, 90)
print(x)

x = numpy.percentile(speed, 100)