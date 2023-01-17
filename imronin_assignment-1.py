from numpy import array

import numpy as np

from scipy.linalg import svd

# define a matrix

W = [[8,5,1,5,6],[5,3,8,7,5],[1,8,10,7,8],[5,7,7,8,9],[6,5,8,9,7]]

b = [1,0,0,0,0]

y0 = [0,1,0,0,0]



y1 = np.dot(W, y0) + b



y2 = np.dot(W, y1) + b



# print(y1)

# print(y2)

# print(round(np.linalg.norm(y1, ord=2), 4))

# print(round((np.linalg.norm(y2, ord=2)/np.linalg.norm(y0, ord=2)), 4))



w, v = np.linalg.eig(W)

# print(round(max(w), 4))



dotW = np.dot(W, W)

w2, v2 = np.linalg.eig(dotW)

# print(round(np.sqrt(max(w2)), 4))



# print(np.linalg.svd(W))





m = [-2, -3, 4, -8, 9]

# print(round(np.linalg.norm(m, ord=1), 4))

# print(round(np.linalg.norm(m, ord=2), 4))

# print(round(np.linalg.norm(m, ord=np.inf), 4))









syMat = [ [ 1, 5, 5 ], [ 3, 5, 4 ], [ 5, 3, 1 ] ]

u, s, vh = np.linalg.svd(syMat)

print(s)

symEig, symEigV = np.linalg.eig(syMat)

print(symEig)



syMatSqr = np.dot(syMat, syMat);

# print(syMatSqr)



symEigSqr, symEigVSqr = np.linalg.eig(syMatSqr)

print(np.sqrt(symEigSqr))