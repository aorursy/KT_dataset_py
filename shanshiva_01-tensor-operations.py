# Import torch and other required modules

import torch
# Example 1 - working (change this)

t1 = torch.ones((5,5), requires_grad=True)

t2 = t1 * 10

t3 = t2.trace()



t3.backward()



print(t1.grad)
# Example 2 - working



input_x = torch.randn((5,5), requires_grad=True)

w = torch.randn((5,5), requires_grad=True)



print(input_x)

print()

print(w)

print()



y = torch.sum(input_x * w)

y.backward()



print(w.grad)

print()

print(input_x.grad)
# Example 3 - breaking (to illustrate when it breaks)

input_x = torch.randn((5,5), requires_grad=True)

w = torch.randn((5,5), requires_grad=True)



y = (input_x * w)

y.backward()



print(w.grad)

print()

print(input_x.grad)
# Example 1 - working

m = torch.rand(4,2)

print(m)

idx = torch.tensor([0,2,4])

n = torch.take(m,idx)

print(n)

# Example 2 - working



m = torch.rand(4,2)

print(m)

idx = torch.arange(3,7,2)

print(idx)

n = torch.take(m,idx)

print(n)
# Example 3 - breaking (to illustrate when it breaks)



import numpy as np



m = torch.rand(4,2)

print(m)

idx1 = np.array([0,2,4])



n = torch.take(m,idx1)



print(n)

# Example 4 - breaking (to illustrate when it breaks)



idx2 = torch.from_numpy(np.array([0,2,4]))



matr = np.linspace((1,2),(10,20),10)

q = torch.take(matr,idx2)

print(q)

# Example 1 - working

mat1 = torch.randn(10, 3)

mat2 = torch.randn(3, 4)



mat3 = torch.matmul(mat1,mat2)



print(mat3)

print()

print(mat3.shape)
# Example 2 - working



mat1 = torch.randn(10, 3, 4)

mat2 = torch.randn(10, 4, 5)



mat3 = torch.matmul(mat1,mat2)



print(mat3)

print()

print(mat3.shape)
# Example 3 - breaking (to illustrate when it breaks)



mat1 = torch.randn(10, 3)

mat2 = torch.randn(10, 4)



mat3 = torch.matmul(mat1,mat2)



print(mat3)

print()

print(mat3.shape)



# Example 1 - working



mat1 = torch.randn(10, 3)



print(mat1)



torch.unsqueeze(mat1, 2)
# Example 2 - working



mat2 = torch.randn(10, 4, 5)



print(mat2)



print(torch.unsqueeze(mat2, -1))
# Example 3 - breaking (to illustrate when it breaks)



mat3 = torch.randn(5, 4)



print(mat3)



torch.unsqueeze(mat1,3)
# Example 1 - working



t1 = torch.tensor([[1, 2, 3],

              [4, 5, 6],

              [7, 8, 9]])



print(t1)



t1.view(9,1)
# Example 2 - working



t1 = torch.randn(5,6)



print(t1)

print()



t2 = t1.view(-1, 5)

print()



print(t2.shape)

print()



print(t2)
# Example 3 - breaking (to illustrate when it breaks)



t3 = torch.tensor([[1, 2, 3],

              [4, 5, 6],

              [7, 8, 9]])



print(t3)

t3 = t3.T

print(t3.view(1,9))