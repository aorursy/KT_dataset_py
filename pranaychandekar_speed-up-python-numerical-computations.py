import numpy as np
u = np.random.rand(50)

u
v = np.random.rand(50)

v
def cosine_similarity(u:np.ndarray, v:np.ndarray):

    assert(u.shape[0] == v.shape[0])

    uv = 0

    uu = 0

    vv = 0

    for i in range(u.shape[0]):

        uv += u[i]*v[i]

        uu += u[i]*u[i]

        vv += v[i]*v[i]

    cos_theta = 1

    if uu!=0 and vv!=0:

        cos_theta = uv/np.sqrt(uu*vv)

    return cos_theta
cosine_similarity(u, v)
!pip show numba
from numba import jit



@jit(nopython=True)

def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):

    assert(u.shape[0] == v.shape[0])

    uv = 0

    uu = 0

    vv = 0

    for i in range(u.shape[0]):

        uv += u[i]*v[i]

        uu += u[i]*u[i]

        vv += v[i]*v[i]

    cos_theta = 1

    if uu!=0 and vv!=0:

        cos_theta = uv/np.sqrt(uu*vv)

    return cos_theta
cosine_similarity_numba(u, v)
k = 10 # Change this value and run the below cells to experiment with different number of computations.
%%timeit

for i in range(k):

    cosine_similarity(u, v)
%%timeit

for i in range(k):

    cosine_similarity_numba(u, v)