import numpy as np
from scipy.special import factorial
def digitsum(number):
    return sum([int(digit) for digit in str(number)])
zehn = int(factorial(10, exact=True))
hundert = int(factorial(100, exact=True))

zehn, hundert
digitsum(zehn), digitsum(hundert)