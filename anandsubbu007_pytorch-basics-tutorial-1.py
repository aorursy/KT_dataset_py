#Importing Library
import torch
import numpy as np
#tuples
lst = ([1,2],[2,3])
print(lst)
print(type(lst))
#tuples --> tensor
l = torch.tensor(lst)
print(l)
print(type(l))
#Numpy --> tensor
l1 = torch.tensor(np.array([1,2,3,4,5]))
print(l1)
print(type(l1))
#Tensor to Numpy
l2 = l1.numpy()
print(l2)
print(type(l2))
#Other Tensor Formates
print("Empty Tensor    :",torch.tensor([]))
print("Float Tensor    :",torch.tensor([1,2,3],dtype=torch.float))
print("Torch of Ones  :\n",torch.ones(3,3))
print("\nTorch of Zeros :\n",torch.zeros(3,2))
print("\nTorch of Randomn Number :\n",torch.rand(2,3))
print("\nTorch with Range of Number :\n",torch.arange(1,5))
print("\nTorch with Linespace of Number :\n",torch.linspace(1,10,6))
print("\nTorch with Identity Matrix :\n",torch.eye(3))
print("\nTorch with complete Matrix :\n",torch.full((2,3),3))
l = torch.arange(8)
v = l.view(2,4)
r = l.reshape(4,2)
print("A tensor of range 7 : ", l)
print("\nWhen we use view    \n",v)
print("\nWhen we use Reshape \n",r)
print("\nFind value in tensor using index of value  :  ",v[1,3].item())
print("\nFind value in tensor using index of value  :  ",r[2,0].item())
# slicing
print("slicing the tensor  :",l[2:6])
print("slicing the tensor  :",l[:4])
x = torch.ones(3,2)
y = torch.full((3,2),3)
print("Tensors of Ones   ,x :\n",x)
print("\nTensors of Full ,y :\n",y)
print("Addition of Tensors      , x+y :\n",x+y)
print("\nSubtraction of Tensors   , x-y :\n",x-y)
print("\nMultiplication of Tensors, x*y :\n",x*y)
print("\nDivition of Tensors      , x/y :\n",x/y)
# IS cuda available in your Device
print("IS cuda available in your Device  ?  : ",torch.cuda.is_available())
print("Count of Available CUDA Device       : ",torch.cuda.device_count())
print("Available Device Name                : ",torch.cuda.get_device_name())
#object for cuda device
cuda0 = torch.device('cuda:0')
print("Device used in this  :  ",type(cuda0))
print("\nSample torch with GPU device  :\n",torch.ones(2,2,device =cuda0))
%%time
for i in range(1000):
  x = np.random.randn(100,100)
  y = np.random.randn(100,100)
  z = x*y
%%time
for i in range(1000):
  x = torch.randn(100,100)
  y = torch.randn(100,100)
  z = x*y
%%time
for i in range(100):
  x = torch.randn(1000,1000,device = cuda0)
  y = torch.randn(1000,1000,device =cuda0)
  z = x*y