from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from scipy.io import loadmat
mnist = loadmat("/kaggle/input/mnist-original/mnist-original")
mnist['label'][0]
from scipy.io import loadmat
mnist = loadmat("/kaggle/input/mnist-original/mnist-original")
mnist_data = mnist["data"].T  # [70,000, 784]
mnist_label = mnist["label"][0]  # [70,000]

# 60000 is number of training examples
train_data = mnist_data[:60000]
train_label = mnist_label[:60000]

test_data = mnist_data[60000:]
test_label = mnist_label[60000:]
# Show one image
import matplotlib.pyplot as plt
plt.imshow(train_data[15000].reshape(28, 28), cmap='gray')
print('Label: ', train_label[15000])
# Make a Dataset object to load data
class DummyDataset(Dataset):
    def __init__(self):
        print('123')
        
    def __getitem__(self, i):
        print('getting item')
        return i
    
    def __len__(self):
        return 10
# Make a Dataset object to load data
class MNISTDataset(Dataset):
    def __init__(self, data, label, transform=None):
        # re-scaling to [0,1] by diving by 255.
        data = data.astype(np.float32)
        label = label.astype(np.int)
        self.data, self.label = data / 255., label
        self.transform = transform
        
    def __getitem__(self, i):
        sample_data, sample_label = self.data[i], int(self.label[i])
        sample_data = sample_data.reshape(28, 28)
        if self.transform:
            sample_data = self.transform(sample_data)
        return sample_data, sample_label
    
    def __len__(self):
        return len(self.data)
# Configurations
lr = 10
batch_size = 32
device = torch.device('cuda:0')
epoch_nb = 5
log_interval = 100
# Initializing dataset
# TODO: try not normalizing and see if that affects performance
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = MNISTDataset(train_data, train_label, transform=transform)
test_ds = MNISTDataset(test_data, test_label, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
# Our model to be used with our custom mean squared error loss function
class StitchedLinearNet(nn.Module):
    def __init__(self):
        super(StitchedLinearNet, self).__init__()
        self.layer1 =  nn.Linear(784, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        # need to flatten x
        bs, channel, width, height = x.shape
        x = x.view(bs, -1)   # TODO: why?
        
        # run through model
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        out = torch.sigmoid(x)
        return out
# our mse_loss
def mse_loss(output, target, reduction='sum'):
    # output: tensor, shape: [batch_size, 10]
    # target: tensor, shape: [batch_size], needs to be turned into one-hot vectors
    # reduction: how to reduce output losses, either 'sum' or 'mean'
    # Output: MSE loss for a batch wrt output and target
    
    bs, output_size = output.shape
    device = output.device
    # turn target into one-hot vectors
    # make dummy zeros and send to the same device as output
    target_onehot = torch.zeros(bs, output_size).to(device)
    
    # naive way of doing this
    # TODO: show how to do this in a matrix-way: with huge spped up
    for i in range(bs):
        target_onehot[i][target[i]] = 1.  # turn it on!
    
    # import pdb; pdb.set_trace()
    
    if reduction == 'sum':
        loss = ((output - target_onehot)**2).sum(dim=-1)
    else:
        loss = ((output - target_onehot)**2).mean(dim=-1)
        
    return loss.mean()
# train loop with mse_loss
def train_mse(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()  # put model to train mode
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output, target, reduction='sum')
        loss.backward()
        #optimizer.step()
        for name, param in model.named_parameters():
            with torch.no_grad():
                # print(name, param.shape)
                # update, SGD style
                param.add_(param.grad, alpha=-lr)
                param.grad.zero_()  # zero out gradients after used
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            losses.append(loss.item())
            
    return losses
# test routine with mse loss
def test_mse(model, device, test_loader):
    model.eval()  # put model in evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # not accumulating gradients
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return test_loss, accuracy
# define model and optimizer
net = StitchedLinearNet()
net = net.to(device)  # send model to device
optimizer = optim.SGD(net.parameters(), lr=lr)
# number of epochs
epoch_nb = 5
# Sanity test at the beginning
test_mse(net, device, test_loader)
 
for epoch in range(epoch_nb):
    # train for 1 epoch
    train_mse(net, device, train_loader, optimizer, epoch, log_interval=log_interval)
    # test at the end of epoch
    test_mse(net, device, test_loader)
# Add more metrics tracking
# Sanity test at the beginning
train_loss, test_loss, test_accuracy = [], [], []

loss, accuracy = test(net, device, test_loader)
test_loss.append(loss)
test_accuracy.append(accuracy)


for epoch in range(epoch_nb):
    # train for 1 epoch
    losses = train_mse(net, device, train_loader, optimizer, epoch, log_interval=log_interval)
    train_loss += losses
    # test at the end of epoch
    loss, accuracy = test_mse(net, device, test_loader)
    test_loss.append(loss)
    test_accuracy.append(accuracy)
plt.figure(figsize=(10, 6))
plt.plot(train_loss)
plt.figure(figsize=(10, 6))
plt.plot(test_loss)
plt.figure(figsize=(10, 6))
plt.plot(test_accuracy)
# Custom Optimizer

lr = 1.
for name, param in net.named_parameters():
    with torch.no_grad():
        print(name, param.shape)
        # update, SGD style
        param.add_(param.grad, alpha=lr)
    
