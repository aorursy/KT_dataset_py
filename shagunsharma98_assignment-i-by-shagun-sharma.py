# Import torch and other required modules

import torch
project_name='01-pytorch basics-5 important functions'
# Example 1 - working (change this)

tens1=torch.tensor([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
print(tens1)
# Example 2 - working

tens2=torch.tensor([[1, 0], [0, 1]])

print(tens2)
# Example 3 - breaking (to illustrate when it breaks)

tens3=torch.tensor([[1, 2], [3, 4, 5]])

print(tens3)
# Example 1 - working

tens1=torch.tensor([[1, 2, 3], [4, 5, 6],[7, 8, 9]])

tens1.size()
# Example 2 - working

torch.tensor([[1, 0, 0], [0, 1, 0]]).size()
# Example 3 - breaking (to illustrate when it breaks)

tens3=torch.tensor([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

tens3.size()
tens3.size(2)
# Example 1 - working

tens1=torch.zeros(3, 3)

print(tens1)
tens1.fill_diagonal_(1)
# Example 2 - working

torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]]).fill_diagonal_(0)
# Example 3 - breaking (to illustrate when it breaks)

tens2=torch.tensor([[[1, 2, 3, 4]],[[5, 6, 7, 8]]])

print(tens2)

tens2.size()
tens2.fill_diagonal_(10)
# Example 1 - working

tens1=torch.tensor([1.7])

print(tens1)

tens1.item()
# Example 2 - working

torch.tensor([2]).item()
# Example 3 - breaking (to illustrate when it breaks)

tens2=torch.tensor([[1, 2, 3],[4, 5, 6]])

print(tens2)

tens2.item()
# Example 1 - working

tens1=torch.tensor([[1.1, 1.2, 1.3],[1.4, 1.5, 1.6]])

print(tens1)

torch.chunk(tens1,2,1)
# Example 2 - working

a,b,c=torch.chunk(torch.tensor([1, 2, 3, 4, 5, 6]),3,0)

a,b,c
# Example 3 - breaking (to illustrate when it breaks)

tens2=torch.tensor([1, 2, 3, 4, 5, 6])

torch.chunk(tens2,3,1)
! pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name)