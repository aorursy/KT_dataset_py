import torch

from torch import nn



input = torch.randn(3, 5, requires_grad=True) # shape [m samples, n classes]

# each element in correct_class_index has to have 0 <= value < n (number of classes)

correct_class_index = torch.tensor([3, 0, 4]) # shape [m samples]



CrossEntropyLoss = nn.CrossEntropyLoss()

LogSoftmax = nn.LogSoftmax(dim=1)

NLLLoss = nn.NLLLoss(reduction="none")



print("input:")

print(input)

print("----------------------------------------------")

print("input after log softmax:")

print(LogSoftmax(input))

print("----------------------------------------------")



output = NLLLoss(LogSoftmax(input), correct_class_index)



print("correct_class_index:")

print(correct_class_index)

print("----------------------------------------------")

print("For each sample, nn.NLLLoss() takes the correct class's value, and the negative of that value would be a positive value:")

print(output)

print("----------------------------------------------")

print("mean of output:")

print(output.mean())
NLLLoss = nn.NLLLoss(reduction="mean")

NLLLoss(LogSoftmax(input), correct_class_index)
assert(CrossEntropyLoss(input, correct_class_index) == NLLLoss(LogSoftmax(input), correct_class_index))