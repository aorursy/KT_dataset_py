from __future__ import print_function

import torch
x=torch.empty(5, 3)

print(x)
x=torch.rand(5,3) # It will create a randomly initialized matrix

print(x)
x=torch.zeros(5,3) 

x

x1=torch.zeros(5,3,dtype=torch.long)# matrix with the datatype long

x1
x1
x=torch.tensor([5.5,3]) # crating tensor from the data, its a direct method

print(x)
# tensor can be created based on the existing tensor. The new command will use the same datatype unless it is defined newly.



x1=x.new_ones(5,3, dtype=torch.double)

x=torch.randn_like(x1, dtype=torch.float)

print(x)
print(x1)
print(x.size())
y=torch.zeros(5,3)

y

print(x+y) # it can also be written as print(torch.add(x,y))
print(torch.add(x,y))
print(x[3,]) # printing the 3rd row; (a,b), here a represent rows and b columns
#converting tensor into numpy and vice versa

a=torch.ones(5)

print(a)
b=a.numpy()

print(b)
a.add_(0.5) # adding 0.5 to a

print(a)

print(b)
import numpy as np

a=np.ones(5)

b=torch.from_numpy(a)

np.add(a,1,out=a)

print(a)

print(b)
# moving tensor is very easy. It can be done by using .to method

if torch.cuda.is_available():

    device = torch.device("cuda")          

    y = torch.ones_like(x, device=device)  

    x = x.to(device)                    

    z = x + y

    print(z)

    print(z.to("cpu", torch.double)) 

    