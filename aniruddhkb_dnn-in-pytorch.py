import torch
import torch.optim as optim
from torch.utils.data import *
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
def load_mnist_train():
    path = '/kaggle/input/digit-recognizer/train.csv'
    raw_data = np.loadtxt(path, delimiter = ',', skiprows = 1)
    raw_X = raw_data [:, 1:]
    raw_Y = raw_data[:, 0]
    return (raw_X, raw_Y)
raw_X, raw_Y = load_mnist_train()
def split_data(raw_X, raw_Y):
    train_X_lst = []
    train_Y_lst = []
    test_X_lst = []
    test_Y_lst = []
    for i in range(raw_X.shape[0]):
        if(i % 2 == 0):
            train_X_lst.append(raw_X[i])
            train_Y_lst.append(raw_Y[i])
        else:
            test_X_lst.append(raw_X[i])
            test_Y_lst.append(raw_Y[i])
    train_X = np.array(train_X_lst)
    train_Y = np.array(train_Y_lst)
    test_X = np.array(test_X_lst)
    test_Y = np.array(test_Y_lst)
    return (train_X, train_Y, test_X, test_Y)
train_X, train_Y, test_X, test_Y = split_data(raw_X, raw_Y)
class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).cuda()
        self.y = torch.from_numpy(y).cuda()
        self.len = self.x.shape[0]
    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    def __len__(self):
        return self.len
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        model = nn.Sequential(nn.Linear(784, 512), nn.ReLU())
        model = nn.Sequential(model, nn.Linear(512, 256), nn.ReLU())
        model = nn.Sequential(model, nn.Linear(256, 64), nn.ReLU())
        model = nn.Sequential(model, nn.Linear(64, 10), nn.Softmax())
        self.model = model
    def forward(self, x):
        return self.model(x)
    def parameters(self):
        return self.model.parameters()
learning_rate = 0.00001
num_epochs = 1000
batch_size = 2048

model = DNN()
model.cuda()
train_dset = Data(train_X, train_Y)
trainloader = DataLoader(train_dset, batch_size = batch_size)
costfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

costs = []

for epoch in range(num_epochs):
    for x, y in trainloader:
        optimizer.zero_grad()
        yhat = model(x.float())
        cost = costfn(yhat, y.long())
        costs.append(cost.item())
        cost.backward()
        optimizer.step()
    print("Cost", epoch," = ", cost.item())
plt.plot(costs)
plt.show()
yhat_test = model(torch.from_numpy(test_X).cuda().float())
costfn(yhat_test, torch.from_numpy(test_Y).cuda().long())
Yhat_test_numpy = yhat_test.detach().cpu().numpy()
Yhat_test = np.argmax(Yhat_test_numpy, axis = 1)
Yhat_test
accuracy = 100*(1 - np.sum((Yhat_test != test_Y).astype(int))/test_Y.shape[0])
accuracy
