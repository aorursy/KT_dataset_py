import time
import random
import math

current_milli_time = lambda: int(round(time.time() * 1000))

# Say that you want to initialize a large array of random numbers
n = 200000

start = current_milli_time()

numbers = []
pdfs = []
for _ in range(n):
    numbers.append(random.uniform(10, 20))

# Then you want to calculate the gaussian probability for each number
for i in range(n):
    mu = 15
    sigma_sq = 5
    pdf = 1/math.sqrt(2 * math.pi * sigma_sq) * math.exp(-(numbers[i] - mu) ** 2 / (2 * sigma_sq))
    pdfs.append(pdf)
    
# You initialize a couple of arrays more of the same size, say b and angle
b = []
angle = []
for _ in range(n):
    b.append(random.randint(3, 6))
    angle.append(random.uniform(0, math.pi))

# And you finally want to add b and the sin of that angle to number
for i in range(n):
    numbers[i] += b[i] * math.sin(angle[i])

end = current_milli_time()

print (end - start)
# numpy methods come a 'size' method where you indicate the size of the matrix.
# For example, to initialize an array of size n from random numbers, all you do is the following:

import numpy as np
numbers = np.random.uniform(10, 20, size=n)
import numpy as np

mu, sigma_sq = 0, 1

start = current_milli_time()

numbers = np.random.uniform(10, 20, size=n)
pdf = 1 / np.sqrt(2 * np.pi * sigma_sq) * np.exp(-(numbers - mu) ** 2 / (2 * sigma_sq))

b = np.random.randint(3,6, size=n)
angle = np.random.uniform(0, math.pi, size=n)

numbers += b * np.sin(angle)

end = current_milli_time()

print (end - start)
# What if I'd like an "n x 5" array?
numbers = np.random.uniform(10, 20, size=(n,5))

# That's it!
# Let's say that we want to create two dimensional arrays and add them.

n = 10000
m = 100

start = current_milli_time()

a = [[random.randint(0,100)] * m for _ in range(n)]
b = [[random.randint(0,100)] * m for _ in range(n)]
c = [[0] * n for _ in range(n)]

for i in range(n):
    for j in range(m):
        c[i][j] = a[i][j] + b[i][j]
        
end = current_milli_time()

print (end - start)
start = current_milli_time()

a = np.random.randint(10, 20, size=(n,m))
b = np.random.randint(10, 20, size=(n,m))

c = a + b
        
end = current_milli_time()

print (end - start)