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
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
def isscalar(num):
    if isinstance(num,generic):
        return True
    else:
        return False
    
print(np.isscalar(3.1))
print(np.isscalar([3.1,3.2]))
print(np.isscalar(False))
x=[1,2,3]
y=[4,5,6]

mul=np.cross(x,y)
print(mul)
x=np.matrix([[1,3],[4,5]])
x
a=x.mean(1)
a
a=[[1,0],[0,1]]
b=[1,2]
np.matmul(a,b)
a=np.array([[1,2],[3,4]])
a.transpose()
import torch
a=torch.Tensor(3,2,1)
print(a.tolist())
print(type(a))
