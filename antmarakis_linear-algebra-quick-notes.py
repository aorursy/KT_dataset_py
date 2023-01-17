import numpy as np
v = np.array([[1, 2, 3]])

v, v.shape
from numpy.linalg import norm

norm(v)
u = np.array([[1, 3, 5]])

np.arccos(np.dot(u, v.T) / (norm(u) * norm(v)))
u / norm(u)
a, b = np.array([[1, 3, 2]]), np.array([[3, -1, 0]])

np.dot(a, b.T)
A = np.array([[1,2,3], [4,5,6], [7,8,9]])

A, A.shape
B = np.random.rand(3,5)

C = np.dot(A, B)

C, C.shape
A = np.random.randint(5, size=(3,3))

B = np.random.randint(5, size=(3,3))

C = np.multiply(A, B)



print(A)

print(B)

print(C)
from numpy.linalg import matrix_rank as rank

rank(A)