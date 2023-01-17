import numpy as np
from scipy.sparse import csc_matrix
dense = np.array([[1, 0, 4], [0, 0, 5], [2, 3, 6]])
dense
val = np.array([1, 2, 3, 4, 5, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
indptr = np.array([0, 2, 3, 6])
csc_matrix((val, indices, indptr), shape=(3, 3)).toarray()
