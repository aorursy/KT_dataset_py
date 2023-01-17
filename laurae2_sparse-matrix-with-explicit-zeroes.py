import numpy as np

from scipy.sparse import lil_matrix

from scipy.sparse import csr_matrix
csr = csr_matrix(np.array([[0, 0],

        [2, 1],

        [np.nan, np.nan]]))
csr.todense()
csr[0, :] = [0, 0]
csr
csr.todense()
csr.tolil()
lil = lil_matrix(np.array([[0, 0],

        [2, 1],

        [np.nan, np.nan]]))
lil[0, :] = [0, 0]
lil
lil[0, :] = [0, 1]
lil
lil.tocsr()
lil.tocsr().nnz