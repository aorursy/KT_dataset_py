# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import OrderedDict

%matplotlib inline
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tqdm import tqdm_notebook

from PIL import Image

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(torch.__version__)
class MNISTTrainDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.labels = np.asarray(self.data.iloc[:, 0])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28, 28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)
    
class MNISTTestDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.labels = np.asarray(self.data.iloc[:, 0])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][0:]).reshape(28, 28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)
train = pd.read_csv('../input/train.csv')
dfs = np.split(train, [5600], axis=0)
valid = dfs[0]
train = dfs[1]
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')

sample.head()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = MNISTTrainDataset(data=train, root_dir='../input', transform=transform)
trainloader =  DataLoader(train_set, batch_size=64, shuffle=True)
valid_set = MNISTTrainDataset(data=valid, root_dir='../input', transform = transform)
validloader = DataLoader(valid_set, batch_size = 64, shuffle=True)
test_set = MNISTTestDataset(data=test, root_dir='../input', transform=transform)
testloader =  DataLoader(test_set, batch_size=64, shuffle=True)
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_sizes[0])),
                                   ('relu1', nn.ReLU()),
                                   ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                                   ('relu2', nn.ReLU()),
                                   ('logits', nn.Linear(hidden_sizes[1], output_size)),
                                   ('output', nn.LogSoftmax(dim=1))]))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
epochs = 5
steps = 0
print_every = 40

for i in range(epochs):
    running_loss = 0
    for ii, (images, labels) in enumerate(trainloader):
        steps += 1
        
        images.data.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(i+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for i in validloader:
        images, labels = i
        images.data.resize_(images.size()[0], 784)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%' % (100 * correct / total))
pred = np.empty((0, 10), float)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
predicted = torch.FloatTensor(0,0)
for i in iter(testloader):
    images, _ = i
    images.data.resize_(images.size()[0], 784)
    outputs = model(images)
    _, temp = torch.max(outputs, 1)
    temp = temp.float()
    print(type(predicted), type(temp))
    predicted = torch.cat((predicted, temp), dim=0)
    print(predicted)
    
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % predicted.numpy().argmax() for j in range(32)))

image

predicted = predicted.numpy()
predicted = predicted.astype(int)
predicted

results = pd.Series(predicted, name='Label')
results = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
results
results.to_csv("mnist_recognizer.csv",index=False)