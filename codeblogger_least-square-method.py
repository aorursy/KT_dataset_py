import numpy as np

import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])

y = np.array([2,4,5,4,5])
x_mean = np.mean(x)

y_mean = np.mean(y)



print("x_mean = ", x_mean)

print("y_mean = ", y_mean)
# x - x_mean = a



a = []

for (i,_x) in enumerate(x):

    a.append(_x - x_mean)



print(a)
# y - y_mean = b



b = []

for (i,_y) in enumerate(y):

    b.append(_y - y_mean)

    

print(b)
# (x - x_mean)^2 = a^2 = c



c = []

for _a in a:

    c.append(_a**2)



print(c)
# d = (x - x_mean)*( y - y_mean ) = a*b



d= []

for (i,_) in enumerate(a):

    d.append(a[i]*b[i])



print(d)
c = np.sum(c)

d = np.sum(d)



b1 = d/c

print(b1)
# y = b0 + b1*x

# 4 = b0 + 0.6*3



b0 = 4 - b1*3

print("b0: ", b0)
print("Equation: y = 2.2 + 0.6*x")