# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

np.arange(10)

np.arange(1,10)
np.arange(1,10,2)
np.arange(1,20,3, dtype=np.float64)
x = np.array([1,3,5,7])
type(x)
np.array([(1,3,5),(7,9,11),(13,15,17)]).ndim
np.zeros((2,3), dtype=np.int16)
np.linspace(2,4,10)
np.random.random((3,3))
threeD = np.arange(1,30,2).reshape(3,5)

print(threeD)
threeD.itemsize
a = np.arange(5)

b = np.array([2,3,4,6,7])

a
diff = a-b

diff
b**2
2*b
np.sin(a)
np.sum(a)
np.max(b)
np.min(b)
b>2
a*b
x = np.array([[1,1],[0,1]])

y = np.array([[2,0],[3,4]])

x*y
x.dot(y)
x.sum()
x.sum(axis=0)
z = np.random.random((3,4))

z
np.mean(z)
np.median(z)
np.std(z)
data_set = np.random.random((2,3))

data_set
np.reshape(data_set, (3,2))
np.reshape(data_set, (6))
np.ravel(data_set)
ds = np.random.random((5,10))

ds
ds[1]
ds[1][1]
ds[1,0]
ds[1:3]
ds[1:3,0]
ds[::2]