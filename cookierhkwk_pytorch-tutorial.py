import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
np.random.seed(1234)

x = torch.empty(5,3)

print(x)
x = torch.rand(5,3)

print(x)
x = torch.zeros(5,3,dtype = torch.long)

print(x)
x = torch.tensor([5.5,3])

print(x)
x = x.new_ones(5, 3, dtype = torch.double)

print(x)

x = torch.randn_like(x, dtype = torch.float)

print(x)



print(x.size())
y = torch.rand(5,3)

result = torch.empty(5,3) # 이거는 미리 그 공간을 만들어 준 경우 사용

print(x+y)

print(torch.add(x,y))

print(torch.add(x,y, out = result))

y.add_(x) # 보통 y를 변화시키는 함수는 add에다가 _ 를 추가해서 사용

print(y)
print(x)

print(x[:,1]) # 모든 행의 1번 인덱싱
x = torch.randn(4,4)

y = x.view(16)

z = x.view(-1,8)

print(x.size(), y.size(), z.size())
a = torch.ones(5)

print(a)
b = a.numpy()

print(b)
a.add_(1)

print(a)

print(b)
a = np.ones(5)

b = torch.from_numpy(a)

np.add(a, 1, out=a)

print(a)

print(b)
import torch

import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    

    def __init__(self):

        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5*5 square convolution

        # kernal

        self.conv1 = nn.Conv2d(1, 6, 5)

        # 1의 32*32의 이미지가 들어가서 (5,5)의 필터를 거쳐서 차원(특징) 6개, 즉 28*28*6의 직육면체가 생김

        self.conv2 = nn.Conv2d(6, 16, 5)

        # 6개의 이미지가 들어가서, (5,5)의 필터를 거쳐서 차원(특징) 16개, 즉 10*10*16 의 직육면체가 생김.

        # an affine operation : y = Wx + b

        self.fc1 = nn.Linear(16 * 5 * 5 , 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

        

    def forward(self, x):

        # Max pooling over a (2, 2) window

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 28*28*6의 직육면체를 (2,2)의 필터로 stride를 2로해서 max pooling 한다.

        # 14*14*6의 직육면체가 튀어 나온다. 각 칸에는 conv1을 거친 특징들이 들어가 있다.

        # If the size is a square you can only sepcify a single number

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # 10*10*16의 직육면체를  (2,2)의 필터로 stride를 2로해서 maxpooling 한다.

        # 5*5*16의 직육면체가 튀어 나온다. 각 칸에는 conv2을 거친 특징들이 들어가 있다.

        x = x.view(-1, self.num_flat_features(x))

        # 밑에 함수를 보면, 이제 size는 배치 사이즈가 첫번째로 나오니까 [1:]로 size를 받아온다.

        # 사이즈를 받아서 fully connected 를 할 준비

        # view = reshape 함수 이다.  결과적으로 -1, 5*5*16의 공간을 생성

        x = F.relu(self.fc1(x))

        # relu 진행

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    

    def num_flat_features(self, x):

        size = x.size()[1:] # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features



net = Net()

print(net)
params = list(net.parameters())

print(len(params))

print(params[0].size())
import torch

import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt





# Hyper-parameters

input_size = 1

output_size = 1

num_epochs = 60

learning_rate = 0.001



# Toy dataset

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 

                    [9.779], [6.182], [7.59], [2.167], [7.042], 

                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)



y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 

                    [3.366], [2.596], [2.53], [1.221], [2.827], 

                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)



# Linear regression model

model = nn.Linear(input_size, output_size)



# Loss and optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  



# Train the model

for epoch in range(num_epochs):

    # Convert numpy arrays to torch tensors

    inputs = torch.from_numpy(x_train)

    targets = torch.from_numpy(y_train)



    # Forward pass

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    

    # Backward and optimize

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    

    if (epoch+1) % 5 == 0:

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



# Plot the graph

predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.plot(x_train, y_train, 'ro', label='Original data')

plt.plot(x_train, predicted, label='Fitted line')

plt.legend()

plt.show()



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')