import numpy as np

import math

import matplotlib.pyplot as plt

 

s = 5000 

checkpoint = s / 5

xs = np.random.random_sample(s) * 2 - 1

ys = np.random.random_sample(s) * 2 - 1

 

c = 0

plt.figure(figsize=(10, 9))

for i in range(1, s):

    x, y = xs[i], ys[i]

    if x ** 2 + y ** 2 <= 1:

        c += 1

        plt.scatter(x, y, c='blue')

    else:

        plt.scatter(x, y, c='red')

 

    if i % checkpoint == 0 :

        print(i, ':MC estimation =' , c/i * 4 )

 

 

print('MC estimation = ' ,c/s * 4 )

print('PI = ', math.pi)



plt.show()
s = 10000000

checkpoint = s / 5

xs = np.random.random_sample(s) * 2 - 1

ys = np.random.random_sample(s) * 2 - 1

 

c = 0

for i in range(1, s):

    x, y = xs[i], ys[i]

    if x ** 2 + y ** 2 <= 1:

        c += 1

 

    if i % checkpoint == 0 :

        print(i, ':MC estimation =' , c/i * 4 )

 

 

print('MC estimation = ' ,c/s * 4 )

print('PI = ', math.pi)
