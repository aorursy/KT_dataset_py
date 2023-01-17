# import pytorch



import torch

import torch.nn as nn

import torch.nn.functional as F
# The inputs

A = torch.Tensor([[1,1,0],[0,1,0],[0,1,0]])

B = torch.Tensor([[0,0,0],[0,1,1],[0,0,1]])



# let's pad the array

dim = len(A) - 1

A = F.pad(A, (dim, dim, dim, dim))



A.shape, B.shape
# unsqueezing creates a extra-dimention along the given dim.



A = A.unsqueeze(dim=0).unsqueeze(dim=0)

B = B.unsqueeze(dim=0).unsqueeze(dim=0)

A.shape, B.shape
A, B
# define the conv layer

conv = nn.Conv2d(

    in_channels=1, 

    out_channels=1,

    kernel_size=B.shape[-1]

) 



# set the `weights` as the value of matrix B

conv.weight.data = B
with torch.no_grad():  # we don't require to store the gradients for now

    result = conv(A)

result
result.max().round()