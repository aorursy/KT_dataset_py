import numpy as np
guardarAca = np.array([1,2,3,4])
guardarAca
type(guardarAca)
guardarAca + guardarAca
guardarAca / guardarAca
guardarAca + 100
np.sqrt(guardarAca)
np.sin(guardarAca)
guardarAca[0]
guardarAca[2]
guardarAca[4]
guardarAca[1:-1]
guardarAca.mean()
guardarAca.max()
guardarAca.sum()
guardarAca.size
data = np.round( np.random.randn(3,5) * 100 )
data
data*2
data.shape
data = np.array([85, 14, 95, 23, 44, 26, 75])
np.sum(data)
np.square(np.sin(data)) + np.square(np.cos(data))
np.ones(3)
np.full([2,5], fill_value=10)
data = np.array([85, 14, 95, 23, 44, 26, 75])
data.min()
data.argmin()
data.ptp() == (data.max() - data.min())
data.mean()
arr2d = np.array([[1, 2, 3],

                  [4, 5, 6],

                  [7, 8, 9]])



arr2d
arr2d[:1]
arr2d[:2, :]
arr2d[1, :2]
arr2d[:2, 1:] = 0

arr2d
arr2d
arr2d < 2
arr2d[arr2d < 2]
arr2d > 0
arr2d[arr2d > 0]