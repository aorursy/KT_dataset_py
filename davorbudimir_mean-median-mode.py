"""
The mean value is the average value.
The median value is the value in the middle.
The mode value is the value that appears the most number of times.
"""

import numpy
from scipy import stats

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

print(x)

x = numpy.median(speed)

print(x)

x = stats.mode(speed)

print(x)

