import numpy as np
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x)
x = x.reshape((6, 1))
print(x)
x = x.reshape((2, 3))
print(x) 