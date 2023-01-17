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
class DenseNetBlock(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super().__init__()
        self.conv_b1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, 4*growth_rate, kernel_size=1, bias=False)
        )
        self.conv_b2 = nn.Sequential(
            nn.BatchNorm2d(4*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        out = self.conv_b1(x)
        out = self.conv_b2(out)
        fin = torch.cat([out, x], dim=1)
        return fin
    
class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_t = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2)
        )
        
    def forward(self, x):
        return self.conv_t(x)

class DenseNet(nn.Module):
    def __init__(self, n_layers, growth_rate=32, reduction_rate=0.5, out_dim=10):
        super().__init__()
        self.growth_rate = growth_rate
        in_channel = 2*self.growth_rate
        self.conv_ini = nn.Conv2d(3, in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool_ini = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
        self.dense_layer1 = self.make_dense_layer(in_channel, growth_rate, n_layers[0])
        in_channel += n_layers[0] * growth_rate
        out_channel = int(math.floor(reduction_rate*in_channel))
        self.trans1 = Transition(in_channel, out_channel)
        in_channel = out_channel

        self.dense_layer2 = self.make_dense_layer(in_channel, growth_rate, n_layers[1])
        in_channel += n_layers[1] * growth_rate
        out_channel = int(math.floor(reduction_rate*in_channel))
        self.trans2 = Transition(in_channel, out_channel)
        in_channel = out_channel

        self.dense_layer3 = self.make_dense_layer(in_channel, growth_rate, n_layers[2])
        in_channel += n_layers[2] * growth_rate
        out_channel = int(math.floor(reduction_rate*in_channel))
        self.trans3 = Transition(in_channel, out_channel)
        in_channel = out_channel

        self.dense_layer4 = self.make_dense_layer(in_channel, growth_rate, n_layers[3])
        in_channel += n_layers[3] * growth_rate
        
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(in_channel, out_dim)
        
    def make_dense_layer(self, in_channel, growth_rate, num_blocks):
        layers= []
        print(num_blocks)
        for _ in range(num_blocks):
            layers.append(DenseNetBlock(in_channel, growth_rate))
            in_channel += growth_rate
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pool_ini(self.conv_ini(x))
        out = self.trans1(self.dense_layer1(out))
        out = self.trans2(self.dense_layer2(out))
        out = self.trans3(self.dense_layer3(out))
        out = self.dense_layer4(out)
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        prob = self.fc(out)
        return prob
def num_parameters(m):
    return sum([paras.numel() for paras in m.parameters() if paras.requires_grad])

OUTPUT_DIM=10
DENSE_LAYER_121 = [6,12,24,16]
model = DenseNet(n_layers=DENSE_LAYER_121)
# init weights
def init_weights(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
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
