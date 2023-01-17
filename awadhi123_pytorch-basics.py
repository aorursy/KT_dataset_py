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
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_name(1) #will hrow error
tensor = torch.randn(2, 2)
tensor.dtype
tensor.device
tensor_moved = tensor.to(torch.device('cuda:0') , dtype=torch.float64)
tensor_moved.dtype, tensor_moved.device
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
src
torch.take(src, torch.tensor([0, 2, 5]))
torch.take(src, torch.tensor([0, 1, 2]))
x = torch.randn(3, 2)
y = torch.ones(3, 2)
x,y
torch.where(x>0,y,x)
torch.where(x>0,x,y)
a = torch.empty(3, 3)
a 
b = torch.empty(3, 3)
b
torch.randint(3, 5, (3,))
torch.randint(3, 5, (3,2))
torch.randn(2, 3)
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'tensor.pt')
loaded = torch.load('tensor.pt')  # loads tensor object from file and stores it on CPU
loaded.device, loaded.dtype
loaded = torch.load('tensor.pt', map_location=lambda storage, loc: storage)
loaded.device, loaded.dtype
loaded = torch.load('tensor.pt', map_location=lambda storage, loc: storage.cuda(0)) #Note : cuda(0) is used because I have only  one GPU that is GPU-0.
# Suppose if we have 2 GPUs and want to load data on GPU-1 then we have to use cuda(1). This concept is also mentioned above.
loaded.device, loaded.dtype
# Map tensors from GPU 1 to GPU 0
#torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
# you can try above code you have multiple GPUs
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
    y = x* 2
y.requires_grad

a = x*3
a.requires_grad  # x's requires_grad property has been inherited by a
torch.no_grad()
s = x*3
d = x*8
s.requires_grad, d.requires_grad
torch.set_grad_enabled(False)
s = x*3
d = x*4
s.requires_grad,d.requires_grad