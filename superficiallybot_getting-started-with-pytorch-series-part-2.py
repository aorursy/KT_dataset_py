import torch

import torch.nn as nn

from torch.autograd import Variable

import numpy as np
# defining x and y values

x_train = [i for i in range(11)]

y_train = [3*i for i in x_train]





x_train = np.array(x_train, dtype = np.float32)

x_train = x_train.reshape(-1, 1)



y_train = np.array(y_train, dtype = np.float32)

y_train = y_train.reshape(-1, 1)
class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LinearRegressionModel, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        

    def forward(self, x):

        out = self.linear(x)

        return out
# hyper-parameters

input_dim = 1

output_dim = 1

epochs = 300

learning_rate = 0.01
model = LinearRegressionModel(input_dim, output_dim)

# if gpu, model.cuda()
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# MODEL TRAINING

loss_list = []

for epoch in range(1, epochs + 1):

    inputs = Variable(torch.from_numpy(x_train))

    labels = Variable(torch.from_numpy(y_train))

    

    optimizer.zero_grad()

    

    outputs = model(inputs)

    

    loss = criterion(outputs, labels)

    

    loss.backward()

    loss_list.append(loss.data)

    optimizer.step()

    

    print('Epoch: {}, loss: {}'.format(epoch, loss.item()))
test_var = Variable(torch.Tensor([[11.0]]))

pred_y = model(test_var)
print('prediction on x = 11 (after training) : ', pred_y.item())
print('Real Prediction as per the relation y = 3x: x = 11 => y = 33')
y_test = np.arange(20,30, dtype = np.float32)

y_test = y_test.reshape(-1, 1)
y_test = Variable(torch.from_numpy(y_test))
preds = model(y_test)
for i in zip(preds, np.arange(20,30)):

    print(f'Prediction when x = {i[1]} -> y = {i[0]}')

    
#import the necessary libraries



import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

import matplotlib.pyplot as plt
sns.lineplot(x = range(10), y = np.array(loss_list[:10]))

plt.show()
sns.scatterplot(x = np.arange(20,30), y = preds.data.numpy().squeeze(), color = 'red', label = 'predicted')

sns.lineplot(x = np.arange(20, 30), y = 3 * np.arange(20,30), color = 'green', label = 'Actual')

plt.show()