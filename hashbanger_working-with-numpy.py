lis = [1,2,3,4,5,6]
import numpy as np
np.array(lis)
myMatrix = [[1,2,3],[4,5,6],[7,8,9]]
np.array(myMatrix)
np.arange(0,10,2)
np.zeros(10)
np.zeros((3,4))
np.ones((3,4))
np.linspace(2,10,23)
np.random.rand(3)
np.random.randn(3)
np.random.randn(3,3)
np.random.randint(10,100,3)
randarr = np.random.randint(10,100,25)
randarr
np.reshape(randarr,(5,5))
randarr.max()
randarr.argmax()
randarr.min()
randarr.argmin()
randarr.shape
ran = randarr.reshape(5,5)
ran
ran.shape
ran.dtype
liss = np.array(list(range(0,24,2)))
liss
liss[7]
liss[:]
liss[5:]
liss[-1:-12:-1]
liss[-1:0:-1]
liss[1:5] = 5
liss
lis = liss[1:5]
lis[:] = 0
lis
liss
lis[0] = 99
lis
liss
lis = liss.copy()
lis
lis[:4] = 32
lis
liss
alist = np.array([list(range(1,26))])
alist = alist.reshape(5,5)
alist
alist[1][1]
alist[1:4,1:4]
blist = np.array(list(range(2,30)))
blist
temp = blist > 9
temp
blist[temp]
blist[blist > 9]
alist = np.array(list(range(0,24,2)))
alist
alist + alist
alist - alist
alist * alist
alist / alist
alist ** 3
alist - 100
np.sqrt(alist)
np.exp(alist)
np.max(alist)
alist.max()
np.sin(alist)
np.log(alist)
