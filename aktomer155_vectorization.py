import numpy as np

import time as time
a = np.random.rand(1000000)

b = np.random.rand(1000000)



tic = time.time()

c = np.dot(a,b)

toc = time.time()

x = toc-tic



print(c)

print("Vectorized Version:", str(x*1000) + "ms")



c = 0

tic = time.time()

for i in range(1000000):

    c += a[i]*b[i]

toc = time.time()

y = toc-tic



print(c)

print("For loop:", str(y*1000) + "ms")



print("Ratio =", y/x)