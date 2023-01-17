# Import torch and other required modules
import torch
# Example 1 - working (change this)
first_tensor = torch.tensor([[1, 2], [3, 4]])
second_tensor = torch.tensor([[3.2, 5]])

result = torch.add(first_tensor, second_tensor)
print(result)
print("Data type of result",result.dtype)
print("Data type of first_tensor",first_tensor.dtype)
print("Data type of second_tensor",second_tensor.dtype)
# Example 2 - working
first_tensor = torch.tensor([[1, 2], [3, 4.]])
second_tensor = torch.tensor([[False, True]])
result = torch.add(first_tensor, second_tensor)
print(result)
print("Result after setting alpha to 2")
result = torch.add(first_tensor, second_tensor, alpha = 2)
print(result)
# Example 3 - breaking (to illustrate when it breaks)
first_tensor = torch.tensor([[1, 2], [3, 4]])
second_tensor = torch.tensor([[3.2, 5, 3]])

result = torch.add(first_tensor, second_tensor)
# Example 1 - working
input_prob = torch.tensor([[0.5, 0.5],[1,0], [0.3, 0.7]])
torch.bernoulli(input_prob)
# Example 2 - working
input_prob = torch.tensor([[0, 0.5, 1]])
torch.bernoulli(input_prob)
# Example 3 - breaking (to illustrate when it breaks)
input_prob = torch.tensor([[0, 0, 1]])
print("Data Type of input probabilities", input_prob.dtype)
torch.bernoulli(input_prob)
# Example 1 - working
torch.randn((3,4))
# Example 2 - working
torch.randn((1), dtype = torch.float32 )

# Example 3 - breaking (to illustrate when it breaks)
torch.randn((1), dtype = torch.long )
# Example 1 - working
torch.dot(torch.tensor([2, 3, 4]), torch.tensor([2, 1, 5]))
# Example 2 - working
torch.dot(torch.tensor([0.0, 2, 3, 4]), torch.tensor([2.0, 5.5, 0, 1]))
# Example 3 - breaking (to illustrate when it breaks)
torch.dot(torch.tensor([2.0, 5.5, 0, 1]), torch.tensor([0, 2, 3, 4]))
# Example 1 - working
input_tensor = torch.tensor([[1, 2], [3, 4]])
torch.rot90(input_tensor, k = 1, dims= [0,1])
# Example 2 - working
input_tensor = torch.tensor([[1, 2], [3, 4]])
torch.rot90(input_tensor, k = 1, dims= [1,0])
# Example 3 - breaking (to illustrate when it breaks)
input_tensor = torch.tensor([[1, 2], [3, 4]])
torch.rot90(input_tensor, k = 1, dims= [1])
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
