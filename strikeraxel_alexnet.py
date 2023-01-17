import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import copy
SEED=1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministics = True
# Calculate Image normalization parameters
ROOT = 'data'
train_data = datasets.CIFAR10(root=ROOT, train=True, download=True)
means = train_data.data.mean(axis=(0,1,2))/255
stds = train_data.data.std(axis=(0,1,2))/255
print(f"Calculated mean for images : {means}")
print(f"Calculated std for images : {stds}")
# Define transform for data augmentation
train_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])
# Create Training, Validation and Testing Dataset
train_data_all = datasets.CIFAR10(root=ROOT, train=True, transform=train_transforms, download=True)
test_data = datasets.CIFAR10(root=ROOT, train=False, transform=test_transforms, download=True)

validation_ratio = 0.1
num_validations_data = int(len(train_data_all) * validation_ratio)
num_train_data = len(train_data_all) - num_validations_data
train_data, valid_data = data.random_split(train_data_all, [num_train_data, num_validations_data])
# We don't want validation data to be augumented
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms
print(f"Number of examples in train data : {len(train_data)}")
print(f"Number of examples in valid data : {len(valid_data)}")
print(f"Number of examples in test data : {len(test_data)}")
# Create data batches for train, valid and test
BATCH_SIZE = 126
train_iterator = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
# Visiualizing data with image and labels
def plot_images(images,labels):
    num_images = len(images)
    rows = int(np.sqrt(num_images))
    cols = int(np.sqrt(num_images))
    fig = plt.figure(figsize=(10,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows,cols, i+1)
        ax.imshow(images[i].permute(1,2,0).cpu().numpy())
        ax.set_title(labels[i])
images, labels = zip(*[ (image, label) for image, label in [train_data[i] for i in range(20)]])
labels = [test_data.classes[i] for i in labels]
plot_images(images, labels)
class AlexNet(nn.Module):
    def __init__(self, num_outputs, dropout_prob):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace= True),
            nn.Conv2d(64, 192 ,3,padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace= True),
            nn.Conv2d(192, 384, 3, padding = 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(384,256 ,3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256 , 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_outputs)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
def num_parameters(m):
    return sum([paras.numel() for paras in m.parameters() if paras.requires_grad])

OUTPUT_DIM=10
DROPOUT_PROB = 0.5
model = AlexNet(OUTPUT_DIM, DROPOUT_PROB)
# init weights
def init_weights(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
model.apply(init_weights)
print(model)
print(f"Number of parameters in model : {num_parameters(model):,}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
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
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
