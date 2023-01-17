# Import torch and other required modules

import torch
# Example 1 - working (change this)

te1 = torch.randint(300, (1, 10), dtype=torch.float).flatten()

print(te1)

te2 = torch.ones(10)

print(te2)

torch.stack((te1, te2))
# Example 2 - working

te1 = torch.randint(15, (3, 5), dtype=torch.float)

print(te1)

te2 = torch.empty(3, 5)

te2 = torch.ones_like(te2)

print(te2)

torch.stack((te1, te2), 2)
# Example 3 - breaking (to illustrate when it breaks)

te1 = torch.randint(15, (3, 5), dtype=torch.float)

print(te1)

te2 = torch.empty(3, 5)

te2 = torch.ones_like(te2)

print(te2)

torch.stack((te1, te2), 3)
# Example 1 - working

val = torch.tensor(77.8)

torch.clamp(val, 55, 80)
# Example 2 - working

val = torch.tensor([[55.8, 120.2], [40, 59.9]])

torch.clamp(val, 45, 120)
# Example 3 - breaking (to illustrate when it breaks)

val = torch.tensor([55.8, 99.2, 75.1, 59.9])

mini = torch.tensor([1,2])

torch.clamp(val, mini, 60)
# Example 1 - working

shape_tensor = torch.arange(0,120,2)

shape_tensor
# Example 2 - working

shape_tensor = torch.arange(-.2, 0.09, .1)

shape_tensor
# Example 3 - breaking (to illustrate when it breaks)

shape_tensor = torch.arange(-1, 1.1, .1)

print(shape_tensor)

print("Elements:", shape_tensor.numel())

assert(shape_tensor.numel() == 20)
test_numbers = torch.arange(11)

pow_tensor = test_numbers.pow(5)

pow_tensor
# Example 1 - working

combi_tensor = torch.arange(10)

print(combi_tensor)

combinations = torch.combinations(combi_tensor)

print("Combinations:", combinations.numel())

combinations
# Example 2 - working

combi_tensor1 = torch.arange(5) + 1

combinations = torch.combinations(combi_tensor1, 3, with_replacement=False)

print("Number of 3-way combinations:", combinations.numel())

combinations
# Example 3 - breaking (to illustrate when it breaks)

stack = torch.stack((combi_tensor, combi_tensor))

print(stack)

combinations = torch.combinations(stack)
# Example 1 - working

torch.randperm(10)
# Example 2 - working

A=torch.tensor(0.)

torch.randperm(10, out=A, requires_grad = True)

print(A)

A.dtype
# Example 3 - breaking (to illustrate when it breaks)

torch.randperm(-3, out=A, requires_grad = True)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()