# Import torch and other required modules
import torch
x = torch.tensor([[1., 2], [3, 4]], requires_grad = True)
pw = x.pow(2)
sm = x.sum()
y = x.pow(2).sum()
print(mx)
y.backward()
print("dy/dx = ", x.grad)
Tensor = torch.tensor([[[1., 1, 1], [2, 2, 2], [3, 3, 3]],
                              [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                              [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])
print("Shape of Tensor =", Tensor.shape)
print(Tensor)
y.new_tensor(Tensor).requires_grad_()
new_tensor = torch.tensor([[[1, 1], [2, 2]],
                           [[3, 3], [4, 4]],
                           [[5, 5], [6, 6]],
                           [[7, 7], [8, 8]],
                           [[9, 9], [0, 0]]])
print(new_tensor.shape)
print(new_tensor)
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
torch.tensor([[1, 2], [3, 4, 5]])
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
