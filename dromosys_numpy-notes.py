import numpy as np
a = np.array([10, 6, -4])
b = np.array([2, 8, 7])
a,b
a + b
a > 0
a + 1
m = np.array([[1, 2, 3], [4,5,6], [7,8,9]]); m
2*m
c = np.array([10,20,30]); c
m + c
np.broadcast_to(c[:,None], m.shape)
np.broadcast_to(np.expand_dims(c,0), (3,3))
c.shape
np.expand_dims(c,0).shape
m + np.expand_dims(c,0)
np.expand_dims(c,1)
c[:, None].shape
m + np.expand_dims(c,1)
np.broadcast_to(np.expand_dims(c,1), (3,3))
m, c
m @ c
xg,yg = np.ogrid[0:5, 0:5]; xg,yg
xg+yg
m,c
m * c
(m * c).sum(axis=1)
n = np.array([[10,40],[20,0],[30,-5]]); n
m @ n
(m * n[:,0]).sum(axis=1)
(m * n[:,1]).sum(axis=1)

