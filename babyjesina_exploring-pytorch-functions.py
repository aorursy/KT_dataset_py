# Import torch and other required modules
import torch
# Example 1 - Initializing a random tensor using randn()
x = torch.randn(3,2)
x
torch.transpose(x,0,1)
# Example 2 
y = torch.rand(4,5)
y
torch.transpose(y,1,1)
# Example 3 - breaking (to illustrate when it breaks)
z= torch.randn(3,2)
z
torch.transpose(z,0)
# Example 1 
x = [1,2,3]
y = [4,5,6]
tensor_x = torch.tensor(x)
tensor_y = torch.tensor(y)
torch.cartesian_prod(tensor_x,tensor_y)
# Example 2 - 
a = [8,9]
b = [6,7]
tnsr_a = torch.tensor(a)
tnsr_b = torch.tensor(b)
c = torch.cartesian_prod(tnsr_a,tnsr_b)
c
# Example 3 - breaking (to illustrate when it breaks)
a = [1,2,3]
b = [1,2.3,3]
tensor_a = torch.tensor(a)
tensor_b = torch.tensor(b)
c = torch.cartesian_prod(tensor_a,tensor_b)
c
# Example 1 - 
a = torch.arange(4.)
torch.reshape(a, (2, 2))
# Example 2 - working
x = torch.randn((1,14))
print(x)
torch.reshape(x,(7,2))
# Example 3 - breaking (to illustrate when it breaks)
x = torch.randn((1,11))
print(x)
torch.reshape(x,(3,4))
# Example 1 - working
tensor_a = torch.tensor([1.,2,1])
torch.histc(tensor_a, bins=4, min=0, max=3)
# Example 2 - working
#tensor_x = torch.tensor([1,2,3])
torch.histc(torch.tensor([1,2.,3]), bins= 4, min=0, max=3)
# Example 3 - breaking (to illustrate when it breaks)
torch.histc(torch.tensor([1,2.,3]), bins= 4, min=4, max=3)
# Example 1 -working
a = torch.rand(1,10)
tensor_a = torch.tensor(a)
torch.trace(a)
# Example 2 - working

x = torch.arange(1., 10.).view(3, 3)
print(x)
torch.trace(x)
# Example 3 - breaking (to illustrate when it breaks)
x = torch.arange(1., 10.).view(3, 5)
print(x)
torch.trace(x)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
