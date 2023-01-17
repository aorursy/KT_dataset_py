import numpy as np # linear algebra
import math

a = np.array([5.0, 4.0, 2.5, 4.5, 5.9, 2.8, 3.5])
m = a.mean()
t = 0
for i in a:
    t += (i - m) * (i - m); 
q = t / a.shape[0]
