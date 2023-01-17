import torch

import torch.nn as nn

import torch.nn.functional as F
class HarshNet(nn.Module):

    def __init__(self):

        super(HarshNet, self).__init__()

        

        self.conv1 = nn.Conv2d(1, 6, 3)

        self.conv2 = nn.Conv2d(6, 16, 3)

        

        self.fc1 = nn.Linear(16*6*6, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

        

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #if size is square - only single number is enough

        x = x.view(-1, self.num_flat_features(x)) #resizing/flatting using view function and udf

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    

    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features       
harshnet5 = HarshNet()
print(net)
params = list(harshnet5.parameters())

print(params)
print(params[0].size())

print(len(params))
inp = torch.randn(1, 1, 32, 32)
out = harshnet5(inp)
print(out)
harshnet5.zero_grad()
out.backward(torch.randn(1, 10))
output = harshnet5(inp)
target = torch.randn(10)

print(target.size())
target = target.view(1, -1) #make it similar size as output

criterion = nn.MSELoss()



loss = criterion(output, target)

print(loss)
harshnet5.zero_grad()



print("Before backprop")

print(harshnet5.conv1.bias.grad)



loss.backward()



print("After backprop")

print(harshnet5.conv1.bias.grad)

lr = 0.01



for f in harshnet5.parameters():

    f.data.sub_(f.grad.data * lr) #in-place subtraction for updating the weights - mul with lr and subtract from current weights
import torch.optim as optim
optimizer = optim.SGD(harshnet5.parameters(), lr=0.01)



optimizer.zero_grad()

output = harshnet5(inp)
loss = criterion(output, target)

loss.backward

optimizer.step() #update the parameters - that's all what optimizer does