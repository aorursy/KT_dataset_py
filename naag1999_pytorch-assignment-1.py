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

# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array.
# As a tool, tensors and tensor algebra is widely used in the fields of physics and engineering. 
# It is a term and set of techniques known in machine learning in the training and operation of deep learning models can be described in terms of tensors.

#Creating specific types of tensor types like zeroes, ones etc

torch.zeros([2,4])
torch.zeros([2,4], dtype = torch.int32)

# adding or removing dtype doesn't affect making the tensor as seen above, it just specifies the type of data used
#similarly a tensor of value 1 can also be formed

torch.ones([2,5], dtype = torch.int32)

# modification and accessing the values in the tensor created
# Step 1: lets start by creating a tensor and assigning it to a variable
x = torch.tensor([1,2,4],[4,7,5])
#Step 2: Now lets access using indexes just like in python
print(x[0][1])

# The error that came above is because we forgot to add another bracket around the data values inserted and hence that was taken as one value
x = torch.tensor([[1,2,4], [4,7,5]])
print(x[0][1])
# To modify a value just use the index position and declare that value to the variable like

x[0][1] = 3
print(x[0][1])
# To create an identity matrix 'eye' method is used as:
torch.eye(3)
# To create a tensor which has 'random normal' values can be used when doing mathematical operations
t = torch.normal(torch.zeros(5,5), torch.ones(5,5))
print(t)
# To create a tensor using random values.
# By default, pytorch.rand() function generates tensor with floating point values ranging between 0 and 1.
t = torch.rand([2,5])
print(t)
# To generate specific value less than 'n' but >1. Muliply the tensor variable with n.
# To fill the whole tensor variable with some value, fill_() method is used
t = torch.Tensor( 5, 5 )
t.fill_(1)
