import warnings

warnings.filterwarnings("ignore")



import torch

from torch.autograd import Variable

import torch.nn.functional as F



x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))

y_data = Variable(torch.Tensor([[0.0],[0.0],[1.0],[1.0]]))



class Model(torch.nn.Module):

    

    def __init__(self):

        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1,1)

    

    def forward(self, x):

        y_pred = F.sigmoid(self.linear(x))

        return y_pred

    

model = Model()



criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



for epoch in range(50000):

    

    y_pred = model(x_data)

    

    loss = criterion(y_pred, y_data)

    

    if epoch%5000 == 0:

        print(epoch, loss.item())

        

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    

test1 = Variable(torch.Tensor([[2.0]]))

test2 = Variable(torch.Tensor([[2.8]]))

    

out1 = model(test1)

out2 = model(test2)

    

print("\nfor input {}, output is {}\n".format(test1.data[0][0], out1.data[0][0] > 0.5))

print("for input {}, output is {}".format(test2.data[0][0], out2.data[0][0] > 0.5))