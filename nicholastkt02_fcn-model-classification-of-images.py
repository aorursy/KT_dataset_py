import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(3*96 * 96 , 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 3) # The 3 here implies that we wisht to have 3 final classifications for our model.
    def forward(self, img):
        #print(img.shape)
        flattened = img.view(-1, 3*96 * 96)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return  F.log_softmax(output)

model = FeedForward()
import torchvision.transforms as transforms

trainTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
data_path = 'Data/assignment_train/'
train_dataset = torchvision.datasets.ImageFolder(
    root = data_path,
    transform = trainTransform)

train_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 64,
    num_workers = 0,
    shuffle = True
)
for i, (images, labels) in enumerate(train_iter):
    if i >= 15:
        break
    plt.subplot(3, 5, i+1)
    imgb = images[0, :,:]
    print(imgb.shape)
    
    plt.imshow(imgb[0,:,:], cmap='gray')
    plt.show()
    print(labels)

        
train_acc_loader = torch.utils.data.DataLoader(train_dataset, batch_size =  64)

def get_accuracy(model, train=False):
    if train:
        data = train_dataset
    correct = 0
    total = 0
    for imgs, labels in train_acc_loader:#torch.utils.data.DataLoader(data, batch_size=64):
        output = model(imgs) # We don't need to run F.softmax
        #print(output)
        #break
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total
import torch.optim as optim

#1. Adam, with lr= 0.0001 or 0.001?

def train(model, data, batch_size=64, num_epochs=1):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.01 or 0.001?

    iters, losses, train_acc = [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        print("EPOCH:",epoch)
        for iteration,(imgs, labels) in enumerate(train_loader):
            model.train()
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            model.eval()
            acc = get_accuracy(model, train=False)
            train_acc.append(acc) # compute training accuracy 
            if iteration%10==0:
                print(acc,loss.item())
            n += 1

    # plotting
    plt.title("Loss performance")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
#print(train_dataset.shape)
#debug = train_dataset[:100,:,:,:]

model = FeedForward()
train(model, train_dataset, num_epochs = 70)
import torchvision.transforms as transforms

trainTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
data_path = 'Data/assignment_test/'
test_dataset = torchvision.datasets.ImageFolder(
    root = data_path,
    transform = trainTransform)

test_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 64,
    num_workers = 0,
    shuffle = True
)
train(model, test_dataset, num_epochs = 50)
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
def get_confusion_matrix(model, data):
    torch.no_grad()
    model.eval()
    preds, actuals = [], []
    
    for imgs, labels in data:
        imgs = imgs.unsqueeze(0)
        out = model(imgs)
        _, predicted = torch.max(out, 1)
        preds.append(predicted)
        actuals.append(labels)
        
    preds = torch.cat(preds).numpy()
    actuals = np.asarray(actuals)
    return confusion_matrix(actuals, preds)
get_confusion_matrix(model, test_dataset)
