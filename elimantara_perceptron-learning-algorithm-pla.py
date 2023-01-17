import numpy as np

d = 2
N = 5

w = np.ones(d+1)
x0 = np.ones(5)
x1 = np.array([22, 25, 30, 23, 24]) # age
x2 = np.array([55000, 65000, 15000, 30000, 35000]) # income
X = np.array(np.column_stack((x0, x1, x2)), dtype='int32')