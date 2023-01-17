import numpy as np
import torch
data_py = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
data_py = data_py.reshape(3, 4)

data_t = torch.as_tensor(data_py, dtype=torch.float32)
from torch.autograd import Variable
var_t = Variable(data_t, requires_grad=True)

calc = torch.mean(var_t * var_t)
print(calc)
calc.backward()
print(var_t.grad)

v2 = Variable(data_t, requires_grad=True)
calc2 = torch.div(v2.pow(2), 4)
print(calc2)
calc2.backward(torch.ones_like(v2))
print(v2.grad)

