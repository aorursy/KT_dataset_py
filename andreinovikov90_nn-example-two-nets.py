import torch

import torch.nn as nn

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,1) #линейные преобразования с одним входным и одни выходным параметром

    def forward(self, x): #метод прямого прохождения сети

        x = self.fc1(x)

        return x
net1 = Net() #инициализируем нейросеть случайным образом

net2 = Net()
from torch.autograd import Variable

inp = Variable(torch.randn(1,1,1))

print(inp)



out1 = net1(inp)

out2 = net2(inp)

print(out1, out2)
import torch.optim as optim

def criterion(out, label): #функция потерь

    return (label - out)**2

optimizer1 = optim.SGD(net1.parameters(), lr=0.01, momentum=0.5) #метод оптимизации

optimizer2 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.5) #метод оптимизации



data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

for epoch in range(20): #определяем, сколько эпох будет обучаться 

    for i, data2 in enumerate(data):

        X, Y = iter(data2)

        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)

        optimizer1.zero_grad()

        optimizer2.zero_grad()

        outputs1 = net1(X)

        outputs2 = net2(X)

        loss1 = criterion(outputs1, Y)

        loss2 = criterion(outputs2, Y)

        loss1.backward() #запуск обратного распространения ошибки

        loss2.backward()

        optimizer1.step()

        optimizer2.step()

    print("Epoch {} - loss1: {} - loss2: {}".format(epoch, loss1.data[0], loss2.data[0]))

    
out1 = net1(inp)

out2 = net2(inp)

print(inp)

print(out1)

print(out2)