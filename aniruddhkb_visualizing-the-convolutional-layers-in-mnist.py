import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch.utils.data import *
import torch.nn as nn
def get_data():
    path = '/kaggle/input/digit-recognizer/train.csv'
    raw_data = np.loadtxt(path, skiprows = 1, delimiter = ',')
    raw_X = raw_data[:,1:]
    raw_Y = raw_data[:,0]
    
    raw_X = raw_X.reshape(-1, 1, 28, 28)
    
    m = raw_X.shape[0]
    
    train_X_lst = []
    train_Y_lst = []
    test_X_lst = []
    test_Y_lst = []
    for i in range(m):
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
    
class Data(Dataset):
    def __init__(self,x, y):        
        self.x = torch.from_numpy(x).float().cuda()
        self.y  = torch.from_numpy(y).long().cuda()
        self.len = x.shape[0]
    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    def __len__(self):
        return self.len
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # The architecture used here is based on LeNet-5.
        #28x28x1 --> 24x24x6 --> 12x12x6
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        #12x12x6 --> 8x8x16 --> 4x4x16-->256
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # 256 --> 120
        self.linear1 = nn.Sequential(nn.Linear(256, 120), nn.ReLU())
        # 120 -->84
        self.linear2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        # 84 -->10
        self.linear3 = nn.Sequential(nn.Linear(84, 10), nn.Softmax())
    def forward(self, x):
        a1 = self.conv1(x)
        a2 = self.pool1(a1)
        a3 = self.conv2(a2)
        a4 = self.pool2(a3)
        a5 = self.flatten(a4)
        a6 = self.linear1(a5)
        a7 = self.linear2(a6)
        a8 = self.linear3(a7)
        return a8, (a1, a2, a3, a4, a5, a6, a7)
        
train_X, train_Y, test_X, test_Y = get_data()
learning_rate = 0.0001
batch_size = 128
num_epochs = 100
model = CNN()
model.cuda()
train_dset = Data(train_X, train_Y)
trainloader = DataLoader(train_dset, batch_size = batch_size)
costfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
costs = []
accuracies = []
epochwise_accuracies = []
for epoch in range(num_epochs):
    this_epoch_accuracies = []
    for x, y in trainloader:
        optimizer.zero_grad()
        yhat, _ = model(x)
        cost = costfn(yhat, y)
        cost.backward()
        optimizer.step()
        costs.append(cost.item())
        yhat_np = np.argmax(yhat.detach().cpu().numpy(), axis = 1)
        y_np = y.detach().cpu().numpy()
        accuracy = 100*(1- np.sum( (yhat_np!= y_np).astype(int))/yhat.shape[0])
        accuracies.append(accuracy)
        this_epoch_accuracies.append(accuracy)
    this_epoch_accuracies = np.array(this_epoch_accuracies)
    mean_accuracy = np.mean(this_epoch_accuracies)
    epochwise_accuracies.append(mean_accuracy)
    print("Epoch:", epoch)
    print("Accuracy = ", mean_accuracy)

plt.plot(costs)
plt.show()
plt.plot(accuracies)
plt.show()
plt.plot(epochwise_accuracies)
plt.show()
test_dset = Data(test_X, test_Y)
testloader = DataLoader(test_dset, batch_size)
test_accuracies = []
l1s, l2s, l3s, l4s, l5s, l6s, l7s = [], [], [], [], [], [], []
for x, y in testloader:
    yhat, intermediates = model(x)
    l1s.append(intermediates[0].detach().cpu().numpy())
    l2s.append(intermediates[1].detach().cpu().numpy())
    l3s.append(intermediates[2].detach().cpu().numpy())
    l4s.append(intermediates[3].detach().cpu().numpy())
    l5s.append(intermediates[4].detach().cpu().numpy())
    l6s.append(intermediates[5].detach().cpu().numpy())
    l7s.append(intermediates[6].detach().cpu().numpy())
    yhat_np = np.argmax(yhat.detach().cpu().numpy(), axis = 1)
    y_np = y.detach().cpu().numpy()
    accuracy = 100*(1- np.sum( (yhat_np!= y_np).astype(int))/yhat.shape[0])
    test_accuracies.append(accuracy)
test_accuracies = np.array(test_accuracies)
test_accuracy = np.mean(test_accuracies)
print("Test accuracy = ", test_accuracy)
l1s = np.concatenate(l1s)
l2s = np.concatenate(l2s)
l3s = np.concatenate(l3s)
l4s = np.concatenate(l4s)
l5s = np.concatenate(l5s)
l6s = np.concatenate(l6s)
l7s = np.concatenate(l7s)

y = test_Y
x = test_X
index = 1
print("The displayed image is:", y[index])
plt.imshow(test_X[index][0], cmap = 'gray')
for i in range(l1s[index].shape[0]):
    plt.imshow(l1s[index][i], cmap = 'gray')
    plt.show()
for i in range(l2s[index].shape[0]):
    plt.imshow(l2s[index][i], cmap = 'gray')
    plt.show()
for i in range(l3s[index].shape[0]):
    plt.imshow(l3s[index][i], cmap = 'gray')
    plt.show()
for i in range(l4s[index].shape[0]):
    plt.imshow(l4s[index][i], cmap = 'gray')
    plt.show()
to_display = l5s[index].reshape(16, 16)
plt.imshow(to_display, cmap = 'gray')
plt.show()
to_display = l6s[index].reshape(10, -1)
plt.imshow(to_display, cmap = 'gray')
plt.show()
to_display = l7s[index].reshape(7, -1)
plt.imshow(to_display, cmap = 'gray')
plt.show()
params = list(model.parameters())
for i in range(len(params)):
    params[i] = params[i].detach().cpu().numpy()
for i in range(params[0].shape[0]):
    plt.imshow(params[0][i][0], cmap = 'gray')
    plt.show()
for i in range(params[2].shape[0]):
    plt.imshow(params[2][i][0], cmap = 'gray')
    plt.show()
plt.imshow(params[4], cmap = 'gray')
plt.show()
plt.imshow(params[6], cmap = 'gray')
plt.show()
plt.imshow(params[8], cmap = 'gray')
plt.show()
last_layer = params[8].reshape(10, 7, 12)
for i in range(last_layer.shape[0]):
    plt.imshow(last_layer[i], cmap = 'gray')
    plt.show()
