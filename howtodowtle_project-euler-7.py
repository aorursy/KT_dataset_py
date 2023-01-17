from math import sqrt
import cProfile
import re
def is_prime(n):
    """
    Returns a boolean: 0 if n is not a prime, 1 if n is a prime.
    """
    if n == 1:
        return False
    else:
        bool_i = [n%2 == 0]
        for i in range(3, int(sqrt(n))+1, 2):
            bool_i.append(n%i == 0)
        if sum(bool_i) == 0:
            return True
        elif n == 2:
            return True
        else:
            return False
def prime_i(i):
    """
    Returns the i-th prime number.
    """
    number = 3
    index = 1
    while index < i:
        if is_prime(number):
            index += 1
        if index == i:
            return number
        number += 2
prime_i(10001)
def ratio(x):
    """
    Returns the ratio of the square root of x to x
    in linear space.
    """
    ratio = (sqrt(x)/2)/x
    return ratio
x = prime_i(10001)
print(f"When checking if {x} is a prime number, we can save {round(1-ratio(x),4)*100} % by only checking numbers until the square root of {x} (and only odd numbers after 2).")