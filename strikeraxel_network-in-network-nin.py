import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random
import copy
SEED =12131
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# get nomalization parameters for images
ROOT='data'
all_train = datasets.CIFAR10(root=ROOT,train=True, download=True)
means = all_train.data.mean(axis=(0,1,2))/255
stds = all_train.data.std(axis=(0,1,2))/255
print(f"Mean of image data : {means}")
print(f"Std of image data : {stds}")
train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])
train_data_all = datasets.CIFAR10(root=ROOT, train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR10(root=ROOT, train=False, transform=test_transform, download=True)

valid_ratio = 0.1
num_valid_data = int(len(train_data_all)*valid_ratio)
num_train_data = len(train_data_all) - num_valid_data
train_data, valid_data = data.random_split(train_data_all, [num_train_data, num_valid_data])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.trainform = test_transform

print(f"Total number of train examples : {len(train_data)}")
print(f"Total number of valid examples : {len(valid_data)}")
print(f"Total nnumber of test examples : {len(test_data)}")
def print_images(images, labels):
    num_images = len(images)
    cols = int(np.sqrt(num_images))
    rows = int(np.sqrt(num_images))
    fig = plt.figure(figsize=(10,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows,cols, i+1)
        ax.imshow(images[i].permute(1,2,0).cpu().numpy())
        ax.set_title(labels[i])
        
images, labels =  zip(*[(image, label) for image,label in [train_data[i] for i in range(20)]])
labels = [test_data.classes[i] for i in labels]
print_images(images, labels)
# create generator function for loading image in batches
BATCH_SIZE = 126
train_iterator = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
class NiN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim=output_dim
        self.classifier = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            # Block 2
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            # Block 3
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, output_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            nn.Dropout(0.5),            
        )
        
    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.shape[0], self.output_dim)
        prob = torch.softmax(x, dim=1)
        return prob
def num_params(model):
    return sum([params.numel()  for params in model.parameters() if params.requires_grad])

OUTPUT_DIM = 10
model = NiN(OUTPUT_DIM)
def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
model.apply(init_weight)
print(model)
print(f"Number of parameters : {num_params(model):,}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
criterion.to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred= model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
