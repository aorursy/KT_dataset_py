# Import torch and other required modules
import torch
import numpy as np
# Example 1 
x = torch.tensor([[3.14]])
y = x.item()

print(x.dtype)
print(type(y))
# Example 2
x = torch.tensor([[1, 2], [3, 4]])
y = x.item()

# Example 1 - 
tensor = torch.zeros((2,), dtype=torch.float64)
x = [[2, 6], [26, 1728]]
y = tensor.new_tensor(x)
print(tensor)
print(y)
print(tensor.shape)
print(y.shape)

tensor = torch.zeros((2,), dtype=torch.float64)
x = [[2, 6], [26, 1729]]
y = tensor.new_tensor(x)
print("Before changing value at x[1][1] : ")
print(y)
x[1][1] = 1.618
print("After changing value at x[1][1] : ")
print("x : ", x)
print("y : ", y)

tensor = torch.zeros((2,), dtype=torch.float64)
x = np.array([[2., 6.], [26., 1729.]])
y = torch.from_numpy(x)
print("Before changing value at x[1][1] : ")
print(y)
x[1][1] = 1.618
print("After changing value at x[1][1] : ")
print("x : ", x)
print("y : ", y)
x = torch.tensor([[11.2, 6.022], [1.380, 6.674]])
y = torch.exp(x)
print(y)
x = torch.tensor([[11.2, 6.022], [1.380, 6.674]])
y = torch.sigmoid(x)
print(y)
x = torch.tensor([1.380, 6.022, 1.602, 6.674])
torch.std_mean(x)
x = torch.tensor([[2,3,6,26], [1.380, 6.022, 1.602, 6.674]])
torch.std_mean(x,1)
x = torch.tensor([2,3,6,26])
torch.std_mean(x)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
