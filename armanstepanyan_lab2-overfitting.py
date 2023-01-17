!pip install torchsummary
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchsummary
from IPython.display import clear_output
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from sklearn.metrics import accuracy_score
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Technical function
def mkdir(path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        print('Directory', path, 'is created!')
    else:
        print('Directory', path, 'already exists!')
        
root_path = 'fmnist'
mkdir(root_path)
download = True
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()
transforms.Compose((transforms.ToTensor()))


fmnist_dataset_train = torchvision.datasets.FashionMNIST(root_path, 
                                                        train=True, 
                                                        transform=train_transform,
                                                        target_transform=None,
                                                        download=download)
fmnist_dataset_test = torchvision.datasets.FashionMNIST(root_path, 
                                                       train=False, 
                                                       transform=test_transform,
                                                       target_transform=None,
                                                       download=download)
train_loader = torch.utils.data.DataLoader(fmnist_dataset_train, 
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(fmnist_dataset_test,
                                          batch_size=10000,
                                          shuffle=False,
                                          num_workers=2)
len(fmnist_dataset_test)
for img, label in train_loader:
    print(img.shape)
#     print(img)
    print(label.shape)
    print(label.size(0))
    break
# Сделаем просто несколько постепенно уменьшающих размерность слоёв с нормализацией
class TinyNeuralNetwork(nn.Module):
    def __init__(self, input_shape=28*28, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(), # This layer converts image into a vector to use Linear layers afterwards
            # Your network structure comes here
            nn.BatchNorm1d(input_shape, affine=False, momentum=0.2),
            nn.Linear(input_shape, 300),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.BatchNorm1d(50, momentum=0.2),
            nn.ReLU(),
            nn.Linear(50, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, inp):       
        out = self.model(inp)
        return out
torchsummary.summary(TinyNeuralNetwork().to(device), (28*28,))
model = TinyNeuralNetwork().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = nn.NLLLoss()
# Model training
def train(model, opt, loss_func):
    model.train(True)
    for i in range(30):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            preds = model.forward(x_batch)
            loss = loss_func(preds, y_batch.long())
            loss.backward()
            opt.step()
train(model, opt, loss_func)
def accuracy(model, dataset):
    for x_batch, y_batch in dataset:
        probs = model.forward(x_batch).detach().numpy()
        argmax = np.argmax(probs, axis=1)
    return accuracy_score(argmax, y_batch.detach().numpy())

print("The accuracy: {}".format(accuracy(model, test_loader)))
class OverfittingNeuralNetwork(nn.Module):
    def __init__(self, input_shape=28*28, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(), # This layer converts image into a vector to use Linear layers afterwards
            # Your network structure comes here
            nn.Linear(input_shape, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, inp):       
        out = self.model(inp)
        return out
torchsummary.summary(OverfittingNeuralNetwork().to(device), (28*28,))
batch_indices = list(range(0, len(fmnist_dataset_train), 10))

train_subset = torch.utils.data.Subset(fmnist_dataset_train, batch_indices)

train_loader_subset = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
# Use code from the seminar
def train_model(model, train_loader, test_loader, loss_fn, opt, n_epochs):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    
    for epoch in range(n_epochs):
        ep_train_loss = []
        ep_train_accuracy = []
        ep_test_loss = []
        ep_test_accuracy = []
        model.train(True)        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            preds = model.forward(X_batch)
            loss = loss_func(preds, y_batch.long())
            loss.backward()
            opt.step()
            ep_train_loss.append(loss.item())
            ep_train_accuracy.append(accuracy_score(np.argmax(preds.detach().numpy(), axis=1), y_batch.detach().numpy()))
        model.train(False)
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model.forward(X_batch)                
                ep_test_loss.append(loss_func(preds, y_batch.long()).item())
                ep_test_accuracy.append(accuracy_score(np.argmax(preds, axis=1), y_batch.detach().numpy()))
        train_loss.append(np.mean(ep_train_loss))
        train_accuracy.append(np.mean(ep_train_accuracy))
        test_loss.append(np.mean(ep_test_loss))
        test_accuracy.append(np.mean(ep_test_accuracy))
    return train_loss, train_accuracy, test_loss, test_accuracy
# Decrease learning rate and take a subset of the whole training data
model = OverfittingNeuralNetwork().to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)
loss_func = nn.NLLLoss()
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, train_loader_subset, test_loader, loss_func, opt, 300)
def draw_plot_accuracy():
    plt.figure(figsize=(15, 10))
    plt.title("Accuracy on epoch dependency")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(train_accuracy, label='train data')
    plt.lot(test_accuracy, label='test data')
    plt.legend()
    plt.show()
def draw_plot_loss():
    plt.figure(figsize=(15, 10))
    plt.title("Loss on epoch dependency")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, label='train data')
    plt.plot(test_loss, label='test data')    
    plt.legend()
    plt.show()
draw_plot_accuracy()
draw_plot_loss()
C
class FixedNeuralNetwork(nn.Module):
    def __init__(self, input_shape=28*28, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(), # This layer converts image into a vector to use Linear layers afterwards
            # Your network structure comes here
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(input_shape, affine=False),
            nn.Linear(input_shape, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.BatchNorm1d(300, affine=False),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            nn.LogSoftmax(dim=1)
            
        )
        
    def forward(self, inp):       
        out = self.model(inp)
        return out
torchsummary.summary(FixedNeuralNetwork().to(device), (28*28,))
model = FixedNeuralNetwork().to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
loss_func = nn.NLLLoss()

train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, reduced_train_loader, test_loader, loss_func, opt, 300)
plt.figure(figsize=(15, 8))
plt.title("Accuracy over epochs (with  normalization)")
plt.xlabel("n_epoch")
plt.ylabel("accuracy")

plt.plot(train_accuracy, label='train')
plt.plot(test_accuracy, label='test')
    
plt.legend()
plt.show()
plt.figure(figsize=(15, 8))
plt.title("Loss over epochs (with  normalization)")
plt.xlabel("n_epoch")
plt.ylabel("logloss")

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
    
plt.legend()
plt.show()