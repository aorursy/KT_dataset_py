import torch
import torch.nn as nn
import torch.nn.functional as fn
import matplotlib.pyplot as plt
data = torch.linspace(-1, 1, 100)
x = torch.unsqueeze(data, dim=1)
y = - x.pow(2) + 0.5 * torch.rand(x.size())
class regression(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(regression, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
reg_net = regression(1, 20, 1)
optimizer = torch.optim.SGD(reg_net.parameters(), lr=0.2)
loss = torch.nn.MSELoss()
plt.ion()
for i in range(200):
    pred = reg_net(x)
    
    loss_value = loss(pred, y)
    
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    
    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pred.data.numpy(), 'g-', lw=5)
        plt.text(0.5, 0, 'Loss={:.4}'.format(loss_value.data.numpy()),
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as fn
import matplotlib.pyplot as plt
n_data = torch.ones(100, 2)

x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)

x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), dim=0).type(torch.FloatTensor) # torch.float32
y = torch.cat((y0, y1),).type(torch.LongTensor) # torch.int64
class Classification(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
cls_net = Classification(2, 10, 2)
optimizer = torch.optim.SGD(cls_net.parameters(), lr=0.02)
loss = torch.nn.CrossEntropyLoss()
plt.ion()
for i in range(100):
    pred = cls_net(x)
    loss_value = loss(pred, y)
    
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    
    if i % 5 == 0:
        plt.cla()
        prediction = torch.max(pred, 1)[1]
        predy = prediction.data.numpy()
        targety = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    c=predy, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((predy == targety).astype(int).sum() / float(targety.size))
        plt.text(1.5, -4, 'Accuracy={:.2}'.format(accuracy),
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()
import torch.utils.data as data
dataset = data.TensorDataset(x, y)
loader = data.DataLoader(dataset=dataset,
                         batch_size=10,
                         shuffle=True,
                         num_workers=2)
plt.ion()
for i in range(10):
    for D, L in loader:
        pred = cls_net(D)
        loss_value = loss(pred, L)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if i % 5 == 0:
            plt.cla()
            prediction = torch.max(pred, 1)[1]
            predy = prediction.data.numpy()
            targety = L.data.numpy()
            plt.scatter(D.data.numpy()[:, 0], D.data.numpy()[:, 1],
                        c=predy, s=100, lw=0, cmap='RdYlGn')
            accuracy = float((predy == targety).astype(int).sum() / float(targety.size))
            plt.text(1.5, -4, 'Accuracy={:.2}'.format(accuracy),
                     fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
        
plt.ioff()
plt.show()
n_data = torch.ones(100, 2)
m_data = torch.ones(100, 2)
m_data[:, 1] = -1

# Make one group of data and mark each data as 0 label.
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)

# Make one group of data and mark each data as 1 label.
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

# Make one group of data and mark each data as 2 label.
x2 = torch.normal(-2 * m_data, 1)
y2 = torch.ones(100) + 1
x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)
net_2 = Classification(2, 10, 3)
optimizer = torch.optim.SGD(net_2.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()   # something about plotting

for t in range(100):
    out = net_2(x)                 # input x and predict based on x
    # must be (1. nn output, 2. target), the target label is NOT one-hotted
    loss = loss_func(out, y)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(
            int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy,
                 fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
