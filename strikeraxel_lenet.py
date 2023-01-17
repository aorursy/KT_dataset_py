import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
SEED = 12312
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
resize_transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, transform=resize_transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=resize_transform, download=True)
VALIDATION_RATIO = 0.15
validation_size = int(len(train_data) * VALIDATION_RATIO)
train_size = len(train_data) - validation_size
train_data, valid_data = data.random_split(train_data, [train_size, validation_size])
print(f"Total Number of data in train : {len(train_data)}")
print(f"Total Number of data in test : {len(test_data)}")
print(f"Total Number of data in valid : {len(valid_data)}")
BATCH_SIZE = 64
train_iterator = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
plt.figure(figsize=(10,10))
i=0
for image, label in train_iterator:
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image[0][0], cmap=plt.cm.binary)
    plt.xlabel(label[0])
    if i > 8:
        break
    i+=1
plt.show()
class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 6,
                               kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6,
                               out_channels = 16,
                               kernel_size = 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
def num_params(model):
    return sum([params.numel() for params in model.parameters() if params.requires_grad])

OUTPUT_DIM = 10
model = LeNet(OUTPUT_DIM)
print(model)
print(f"Total number of parameters : {num_params(model)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model.to(device)
criterion.to(device)
def compute_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float()/y.shape[0]
    return acc
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for image, label in iterator:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y_pred = model(image)
        loss = criterion(y_pred, label)
        acc = compute_accuracy(y_pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc +=  acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss= 0
    epoch_acc = 0
    with torch.no_grad():
        for image, label in iterator:
            image = image.to(device)
            label = label.to(device)
            y_pred = model(image)
            loss = criterion(y_pred, label)
            acc = compute_accuracy(y_pred, label)
            epoch_loss += loss.item()
            epoch_acc +=  acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCH = 5
best_loss = float("inf")
for epoch in range(EPOCH):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator ,optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, train_iterator , criterion)
    end_time = time.time()
    
    mins, secs = epoch_time(start_time, end_time)
    if valid_loss < best_loss:
        best_loss = valid_loss 
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {mins}m {secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
model.load_state_dict(torch.load('model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
