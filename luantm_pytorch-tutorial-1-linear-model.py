import torch
from torch.autograd import Variable
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]] ))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]] ))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #one input and on
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = Model()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if epoch % 50 == 0:
        print('epoch {}: LOSS = '.format(epoch), loss.data[0])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
hour_var = Variable(torch.Tensor([[4.0]]))
print('predict (after training) ', 4, model.forward(hour_var))
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]] ))
y_data = Variable(torch.Tensor([[0.], [0.], [1.0], [1.0]] ))
import torch.nn.functional as F
class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
ligistic_model = LogisticModel()
criterion = torch.nn.BCELoss(size_average=True) #Binary cross entroy
optimizer = torch.optim.SGD(ligistic_model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = ligistic_model(x_data)
    loss = criterion(y_pred, y_data)
    if epoch % 100 == 0:
        print(epoch, loss.data)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
hour_var = Variable(torch.Tensor([[1.]]))
print("predict 1h ", ligistic_model(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[5.]]))
print("predict 5h ", ligistic_model(hour_var).data[0][0] > 0.5)
