# Import torch and other required modules

import torch
# Example 1 - working 

import torch

x=torch.rand(2,2)

print(x)

print(x[1,1].item())
# Example 2 - working

import torch

x=torch.zeros([1])

print(x.item())
# Example 3 - breaking (to illustrate when it breaks)

import torch

x=torch.tensor([1, 2])

print(x)

print(x.item())
# Example 1 - working

import torch

x=torch.empty(2,3,3,2)

print(x)
# Example 2 - working

import torch

y=torch.empty(1)

print(y)
# Example 3 - breaking (to illustrate when it breaks)

import torch

z=torch.empty()

print(z)

# Example 1 - working

import torch

import numpy as np



a=np.ones(5)

print(a)

p=torch.from_numpy(a)

print(p)
# Example 2 - working

import torch

import numpy as np



a=np.ones(5)

print(a)

p=torch.from_numpy(a)

print(p)

p.add_(1)

print(a)

print(p)
# Example 3 - breaking (to illustrate when it breaks)

a=np.ones(5)

print(a)

p=torch.from_numpy()

print(p)
# Example 1 - working

import torch



x=torch.rand(4*4)

print(x)

y=x.view(16)

print(y)
# Example 2 - working

import torch



x=torch.rand(4*4)

print(x)

y=x.view(-1,8)

print(y)
# Example 3 - breaking (to illustrate when it breaks)

import torch



x=torch.rand(4*4)

print(x)

y=x.view(15)

print(y)
# Example 1 - working

import torch

mat1=torch.randn(2,2)

mat2=torch.randn(2,2)

print(mat1)

print(mat2)

mat3=torch.matmul(mat1, mat2)

print(mat3)
# Example 2 - working

import torch



tensor1 = torch.ones((1,), dtype=torch.int32)

print (tensor1)

tensor2 = tensor1.add_(1)

print (tensor2)

tensor3=torch.matmul(tensor1,tensor2)

print(tensor3)
# Example 3 - breaking (to illustrate when it breaks)

import torch

mat1=torch.randn(2,2)

mat2=torch.randn(3,2)

print(mat1)

print(mat2)

mat3=torch.matmul(mat1, mat2)

print(mat3)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()