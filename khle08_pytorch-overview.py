import torch
import numpy as np
# Method No.1
d1 = np.arange(12)
T1 = torch.Tensor(d1)
print(d1)
print(T1)
print(T1.dtype)
# Method No.2
d2 = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
t2 = torch.tensor(d2, dtype=torch.int32)
print(d2)
print(t2)
print(t2.dtype)
# Method No.3
d3 = np.random.randint(0, 5, 12)
t3 = torch.as_tensor(d3)#, dtype=torch.int32)
print(d3)
print(t3)
print(t3.dtype)
# Method No.4
d4 = np.random.randint(5, 10, 12)
t4 = torch.from_numpy(d4)
print(d4)
print(t4)
print(t4.dtype)
d1[3] = 100
print(d1, '\n', T1)
d2[1][3] = 100
print(d2, '\n', t2)
d3[4] = 1000
print(d3, '\n', t3)
t3[3] = 1000
print(d3, '\n', t3)
d4[10] = 1000
d4 = d4.reshape(3, 4)
print(d4, '\n', t4)
from torch.autograd import Variable
v1 = Variable([1, 2, 3])
v1 = Variable(t3, requires_grad=True)
print(v1)

t_v1 = v1.data
print(t_v1)

org = t_v1.numpy()
print(org)
eg = torch.tensor([2, 4, 6])
eg_add = eg.add(10)
print(eg_add)
print(eg is eg_add)
eg_add_2 = eg.add_(10)
print(eg_add)
print(eg is eg_add_2)
