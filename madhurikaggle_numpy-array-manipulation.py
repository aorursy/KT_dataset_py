#import numpy library.

import numpy as np
A = np.ones((3, 3))

B = np.zeros((3, 3))
#vstack

np.vstack((A, B))
np.vstack((B, A))
np.hstack((A, B))
np.hstack((B, A))
a = np.array([1, 2, 3])

b = np.array([2, 3, 4])

c = np.array([5, 6, 7])
#column_stack

np.column_stack((a, b, c))
#if we change the order of an arrays.

np.column_stack((b, a, c))
#row_stack

np.row_stack((a, b, c))
np.row_stack((b, c, a))
A = np.arange(16).reshape((4, 4))

A
# horizontal splitting using hsplit.

[B, C] = np.hsplit(A, 2)
B
C.shape
#vertical split using vsplit.

[B, C] = np.vsplit(A, 2)
B
C
C.shape
A
[A1, A2, A3] = np.split(A, [1, 3], axis = 1)
print(A1)
print(A2)
print(A3)
#you can do the same thing by row.

[A1, A2, A3] = np.split(A, [1, 3], axis = 0)
print(A1)
print(A2)
print(A3)