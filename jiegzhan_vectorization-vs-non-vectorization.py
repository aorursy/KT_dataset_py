import numpy as np # linear algebra
import time
# Two matrices with 10 million dimensions
a = np.random.rand(10000000)
b = np.random.rand(10000000)
# Vectorized
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print('Vectorized version calculation result: ',c)
print('Vectorized version process time: '+ str((toc-tic)*1000)+ ' ms')
# Non-vectorized
c = 0
tic = time.time()
for i in range(len(a)):
    c += a[i]*b[i]
toc = time.time()
print('Non-vectorized version calculation result: ',c)
print('Non-vectorized version process time: '+ str((toc-tic)*1000)+ ' ms')
