import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import time
import copy

%matplotlib inline

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# prepare & normalize data and labels
train_data = torch.tensor(df_train.drop(['label'], axis=1).values.reshape((-1,28,28)).astype('float32')) / 255
labels = torch.tensor(df_train['label'].values.astype(np.float32)).long()
test_data = torch.tensor(df_test.values.reshape((-1,28,28)).astype('float32')) / 255

training_dataset = torch.utils.data.TensorDataset(train_data, labels)
# split and prepare the data to be loaded
train_size = int(0.8 * len(training_dataset)) # 80% of the dataset for training
test_size = len(training_dataset) - train_size # remaining 20% of the dataset for testing
train_dataset, test_dataset = torch.utils.data.random_split(training_dataset, [train_size, test_size])
# define batch size and dataloaders
batch_size = 64

dataloaders = dict()
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()

print('Batch shape: ', images.shape)
print('Batch type: ', type(images))
# plot some random images and labels
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(str(labels[idx].item()))
# view an image in more detail
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
# define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()
print(model)
# specify criterion
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train the model
mean_train_losses = []
mean_test_losses = []
test_accuracy_list = []
num_epochs = 25

since = time.time()

for epoch in range(num_epochs):
    model.train()
    
    train_losses = []
    test_losses = []
    for i, (images, labels) in enumerate(dataloaders['train']):
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders['test']):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    mean_train_losses.append(np.mean(train_losses))
    mean_test_losses.append(np.mean(test_losses))
    
    accuracy = 100*correct/total
    test_accuracy_list.append(accuracy)
    print('Epoch {} - Training Loss : {:.4f}, Testing Loss : {:.4f}, Test Accuracy : {:.2f}%'\
         .format(epoch+1, np.mean(train_losses), np.mean(test_losses), accuracy))
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
results = []
for inputs in test_data:
    with torch.no_grad():
        output = model.forward(torch.tensor(inputs))
        ps = torch.exp(output)
        results = np.append(results, ps.topk(1)[1].numpy()[0])
results = results.astype(int)
index = [x+1 for x in df_test.index.tolist()]
df = pd.DataFrame({'ImageId': index, 'Label':results})
df.to_csv("submission.csv", index = False)