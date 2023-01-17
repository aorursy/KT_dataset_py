# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
a=np.array(5)
a.shape
type(a+6)
v=np.array([1,2,3])

v
v.shape
p=np.array([[1,2,3],
        [4,5,6],
        [7,8,9]])
p[0][1]
p.shape
T=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])

T
T.shape

T.reshape(3,5)
tmp=np.array([[2,3,4],[5,6,7]])

tmp
tmp.shape
tmp.reshape(6,1)
tmp+5
tmp.T
ab=(np.random.rand(3,3))
ab
np.matmul(ab,tmp.T)
np.linspace(5,1,4)
np.arange(0,10,0.1)
type(np.arange(0,10,dtype=float)[2])
np.zeros((15,15))
np.ones((5,5))*15
np.linspace(0,10,3)
np.linspace(0,10,50)
np.eye(5)
tmp.flatten()
tmp
np.matmul(tmp,np.eye(3))
np.random.rand(2)
np.random.randn(5,5)
A=np.random.randint(1,100,100000000)
A.max()
A.min()