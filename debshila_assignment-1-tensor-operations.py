# Import torch and other required modules
import torch
!pip install jovian
import jovian
# Example 1 - working
torch.manual_seed(1)
# Example 2 - working
torch.manual_seed(2.5)
# Example 3 - not working
torch.manual_seed('abc')
# Example 1 - creates a vector
torch.randn(5) 
# Example 2 - creates a 3-dimensional array
torch.randn(2,3,4)
# Example 3 - breaking (to illustrate when it breaks)
torch.randn([[1, 2], [3, 4, 5]])
jovian.commit(project = '01-tensor-operations')
# Example 1 - working
t1 = torch.randn(5)
t2 = torch.randn(5)
product = torch.matmul(t1, t2)
print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\nproduct: ',product)
# Example 2 - working
t1 = torch.randn(2,3)
t2 = torch.randn(3,2)
product = torch.matmul(t1, t2)
print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\nproduct: ',product)
# Example 3 - breaking (to illustrate when it breaks)
t1 = torch.randn(2,3)
t2 = torch.randn(2,3)
product = torch.matmul(t1, t2)
print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\nproduct: ',product)
# Example 1 - working
t1 = torch.randn(2)
t2 = torch.randn(2)
dot_product = torch.dot(t1, t2)

print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\ndot_product: ',product)
# Example 2 - working
t1 = torch.randn(5)
t2 = torch.randn(5)
dot_product = torch.dot(t1, t2)

print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\ndot_product: ',product)
# Example 3 - breaking (to illustrate when it breaks)
t1 = torch.randn(5)
t2 = torch.randn(3)
dot_product = torch.dot(t1, t2)

print('tensor 1: ', t1, '\ntensor 2: ', t2,
      '\ndot_product: ',product)
jovian.commit(project = '01-tensor-operations')
# Example 1 - working
t1 = torch.rand(2, 2)
t1_inverse = torch.inverse(t1)
t1_x_t1_inverse = torch.matmul(t1, t1_inverse)
print('t1 mat: ', t1, '\nt1_inverse:', t1_inverse, '\nt1_x_t1_inverse: ',t1_x_t1_inverse)
# Example 2 - working
t1 = torch.rand(4, 4)
t1_inverse = torch.inverse(t1)
t1_x_t1_inverse = torch.matmul(t1, t1_inverse)
print('t1 mat: ', t1, '\nt1_inverse:', t1_inverse, '\nt1_x_t1_inverse: ',t1_x_t1_inverse)
# Example 3 - breaking (to illustrate when it breaks)
t1 = torch.rand(4, 3)
t1_inverse = torch.inverse(t1)
jovian.commit(project = '01-tensor-operations')
