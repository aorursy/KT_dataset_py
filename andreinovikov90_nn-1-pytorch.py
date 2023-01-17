import torch
w = torch.rand(4, 3, 2) #создаем случайный тензор размера 4, 3, 2

x = torch.Tensor(4, 3, 2) #создаем тензор размера 4, 3, 2

y = torch.ones(4, 3, 2) #создаем тензор, заполненный единицами, размера 4, 3, 2

z = torch.zeros(4, 3, 2) #создаем тензор, заполненный нулями, размера 4, 3, 2

print(w)

print(x)

print(y)

print(z)
import torch.nn as nn



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,1) #линейные преобразования с одним входным и одни выходным параметром

    def forward(self, x): #метод прямого прохождения сети

        x = self.fc1(x)

        return x

    

net = Net() #инициализируем нейросеть случайным образом

print(net)

print(list(net.parameters()))
from torch.autograd import Variable

inp = Variable(torch.randn(1,1,1))

print(inp)
out = net(inp)

print(out)
import torch.optim as optim

def criterion(out, label): #функция потерь

    return (label - out)**2

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5) #метод оптимизации



data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

for epoch in range(20): #определяем, сколько эпох будет обучаться 

    for i, data2 in enumerate(data):

        X, Y = iter(data2)

        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)

        optimizer.zero_grad()

        outputs = net(X)

        loss = criterion(outputs, Y)

        loss.backward() #запуск обратного распространения ошибки

        optimizer.step()

    print("Epoch {} - loss: {}".format(epoch, loss.data[0]))
out = net(inp)

print(inp, out)