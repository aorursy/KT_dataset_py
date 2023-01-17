import torch
# create some tensors here
my_tensor = ____ # maybe try torch.rand((3, 4))?
print(my_tensor)
# try printing out some attributes of a tensor
print(____)
# try to do some operations
print(torch.ones(3, 3) * 4)
print(torch.ones(3, 3) + torch.eye(3))
# try
print(torch.ones(2, 3).T)
print(torch.eye(2).reshape(1, 4))
print(torch.eye(4).reshape(2, -1))
print(torch.cat([torch.ones(2, 3), torch.zeros(1, 3)], dim=0))
pass
# try indexing!
print(____)
pass
pass