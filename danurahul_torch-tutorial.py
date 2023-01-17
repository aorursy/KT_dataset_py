# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
torch.__version__
lst=[3,4,5,6]

arr=np.array(lst)
arr
arr.dtype
tensors =torch.from_numpy(arr)

tensors
tensors[:2]
tensors[1:4]
tensors[3]=100
tensors
arr
tensor_arr=torch.tensor(arr)
tensor_arr
tensor_arr[3]=17

tensor_arr
arr
torch.zeros(2,3,dtype=torch.float32)
torch.ones(4,4,dtype=torch.float64)
x=torch.tensor(np.arange(0,12).reshape(4,3))
x
x[:1]
x[:,:1]
a=torch.tensor([1,2,3],dtype=torch.float32)

b=torch.tensor([3,4,5],dtype=torch.int64)
print(a+b)
a
b
torch.add(a,b)
c=torch.zeros(3)
c
torch.add(a,b,out=c)
c
torch.add(a,b).sum()
x=torch.tensor([1,2,3],dtype=torch.float32)

y=torch.tensor([4,5,6],dtype=torch.float32)
print(x*y)
torch.mul(x,y)
x=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float64)

y=torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float64)
torch.matmul(x,y)
torch.mm(x,y)
x@y