import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
def load_data():
    data_path = "/kaggle/input/cifar-10-training-set-csv/train_images.csv"
    labels_path = "/kaggle/input/cifar-10-training-set-csv/train_labels_numeric.csv"
    
    raw_X = np.loadtxt(data_path, delimiter = ',', skiprows = 0).reshape(-1, 32, 32, 3)
    raw_Y = np.loadtxt(labels_path, delimiter = ',', skiprows = 1)[:, 1]
    
    return (raw_X, raw_Y)
raw_X, raw_Y = load_data()
raw_X = raw_X.reshape(-1, 32, 32, 3)
redone_X = np.zeros((raw_X.shape[0], 3, 32, 32))
for i in range(3):
    redone_X[:,i,:,:] = raw_X[:,:,:,i]
raw_X.shape
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
    return train_X, train_Y, test_X, test_Y
train_X, train_Y, test_X, test_Y  = split_data(redone_X, raw_Y)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 1)
        self.model = nn.Sequential(self.model,nn.Conv2d(in_channels = 6, out_channels = 6, kernel_size = 3, stride = 1, padding = 1), nn.Flatten())
        self.model = nn.Sequential(self.model, nn.Linear(32*32*6, 6400), nn.ReLU())
        self.model = nn.Sequential(self.model, nn.Linear(6400, 1024), nn.ReLU())
        self.model = nn.Sequential(self.model, nn.Linear(1024, 256), nn.ReLU())
        self.model = nn.Sequential(self.model, nn.Linear(256, 32), nn.ReLU())
        self.model = nn.Sequential(self.model, nn.Linear(32, 10), nn.Softmax())
    def forward(self, x):
        return self.model(x)
    def parameters(self):
        return self.model.parameters()
class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).cuda().float()
        self.y = torch.tensor(y).cuda().long()
        self.len = self.x.shape[0]
    def __getitem__(self, i):
        return(self.x[i], self.y[i])
    def __len__(self):
        return self.len
lr = 0.0001
epochs = 1000
batch_size = 2048
train_dset = Data(train_X, train_Y)
trainloader = DataLoader(train_dset, batch_size = batch_size)
model = CNN()
model.cuda()
costfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
costs = []
for epoch in range(epochs):
    for x, y in trainloader:
        optimizer.zero_grad()
        yhat = model(x)
        cost = costfn(yhat, y)
        cost.backward()
        optimizer.step()
        costs.append(cost.item())
    print("Cost", epoch, "=", cost.item())
plt.plot(costs)
plt.show()
train_Yhat_one_hot = model(torch.tensor(test_X).float().cuda()).detach().cpu().numpy()
train_Yhat = np.argmax(train_Yhat_one_hot, axis = 1)
train_accuracy = 100*(1 - np.sum((train_Yhat != test_Y).astype(int))/test_X.shape[0])
train_accuracy
test_X
