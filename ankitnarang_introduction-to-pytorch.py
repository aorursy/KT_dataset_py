#!pip install torch
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import torch



import helper
x = torch.rand(3, 2)

x = torch.rand(4,5)

x
y = torch.ones(x.size())

y = torch.zeros(x.size())

y

z = x + y

z
z[0][0]
z[:, 1:4]
# Return a new tensor z + 1

z.add(1)
# z tensor is unchanged

z
# Add 1 and update z tensor in-place

z.add_(1)
# z has been updated

z
z.size()

z.resize_(2, 6)
z
a = np.random.rand(4,3)

a
b = torch.from_numpy(a)

b
b.numpy()
# Multiply PyTorch Tensor by 2, in place

b.mul_(2)
# Numpy array matches new values from Tensor

a