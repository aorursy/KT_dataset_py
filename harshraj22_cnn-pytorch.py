# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing libraries

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Create a transformer to convert image to tensors/ numpy arrays
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
# Loading dataset

# label is determined by the subfolder in which the images are stored
# https://stackoverflow.com/a/54528674/10127204

# For now, Let's import the dataset as it is (without transforming it)
# Edit: 
#     If not transformed, the images are 'pil' and not numpy or tensors, causing error while displaying
# Read here: https://discuss.pytorch.org/t/image-file-reading-typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-class-pil-image-image/9909

dataset = datasets.ImageFolder('../input/cnn-img-data/tiny-imagenet-200/train/', transform=transform)
# See dataset detials
# There are 200 class and each class has 500 images

len(dataset), len(dataset)/500
# Create a dataloader that outputs data in given batch size

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)
# For visualizing images

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
dataIter = iter(dataloader)
# check 3 images, note images are already shuffled
for _ in range(3):
    images, label = next(dataIter)
    arr = np.squeeze(images[0].numpy()[0])
    plt.imshow(arr)
    plt.show()
    print(label[0], end=' ')
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
# try connecting to GPU

# while training, we will enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
resnetModel = models.resnet50(pretrained=True).to(device)
# set all params of resnetModel to 'non-trainable' 
for param in resnetModel.parameters():
    param.requires_grad = False
resnetModel
# Create last layer of resnetModel


# normalize = 

resnetModel.fc = nn.Sequential(
    nn.Linear(2048, 300),
    nn.ReLU(inplace=True),
    nn.Linear(300, 200)
    
).to(device)
# set criteria

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnetModel.fc.parameters())
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

#         for phase in ['train', 'validation']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

#             if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if (index % 100 == 0):
                print(f'index: {index} epoch: {epoch+1}')

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)

        print('{} loss: {:.4f}, acc: {:.4f}'.format('train',
                                                        epoch_loss,
                                                        epoch_acc))
    return model
print('Reached')
model_trained = train_model(resnetModel, criterion, optimizer, num_epochs=20)
# save model
filename_pth = 'cnn_resnet_trained_8.pth'
torch.save(model_trained.state_dict(), filename_pth)
validationDataset = datasets.ImageFolder('../input/cnn-img-data/tiny-imagenet-200/val/', transform=transform)
len(validationDataset), 'validation................'
valDataloader = torch.utils.data.DataLoader(validationDataset, batch_size=32, shuffle=True, num_workers=10)
def eval_val(model, criterion, optimizer, num_epoch=3):
    for epoch in range(num_epoch):
        print(f'epoch {epoch}/ {num_epoch}', '*' * 10, sep = '\n')
        model.eval()
        cur_loss, cur_correct = 0.0, 0.0
        for index, (inputs, labels) in enumerate(valDataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if (index % 30 == 0):
                print(f'index: {index}')
            
            _, pred = torch.max(outputs, axis=1)
            cur_loss += loss.item() * inputs.size(0)
            cur_correct += torch.sum(pred == labels.data)
            
        epoch_loss = cur_loss / len(valDataloader)
        epoch_accu = cur_correct.double() / len(valDataloader)
            
        print(f'Loss: {epoch_loss:.3f}, Acc: {epoch_accu:.3f}')
        
    return model
model_tested = eval_val(resnetModel, criterion, optimizer, num_epoch=1)
print('Now prediction..............')
# save model
# filename_pth = 'cnn_resnet.pth'
# torch.save(model_tested.state_dict(), filename_pth)
# load saved model


# model = models.resnet50(pretrained=False).to(device)
# model.fc = nn.Sequential(
#     nn.Linear(2048, 300),
#     nn.ReLU(inplace=True),
#     nn.Linear(300, 200)
# ).to(device)
# model.load_state_dict(torch.load('./cnn_resnet.pth'))


predDataset = datasets.ImageFolder('../input/cnn-img-data/tiny-imagenet-200/test/', transform=transform)

predDataloader = torch.utils.data.DataLoader(predDataset, batch_size=32, shuffle=False, num_workers=10)
len(predDataset), predDataset[0][0]

predDataloader, len(predDataloader)
predList = []
ano = []

# print(next(iter(predDataloader)))
for index, (imgs, label) in enumerate(predDataloader):
    print(f'Working, {index}')
    with torch.no_grad():
        imgs = imgs.to(device)
        label = label.to(device)
        output = model(imgs)
        
        pred = torch.argmax(output, dim=1)
        predList += [p.item() for p in pred]
        ano += [(p.item(), n) for p, n in zip(pred, label)]
        
        if (index % 100 == 0):
            print(F.softmax(output, dim=1))
#     print(len(imgs), type(imgs))
print('done', len(predList), predList[:5])
# predList
di = {}
li = []

for index, name in enumerate(os.listdir('../input/cnn-img-data/tiny-imagenet-200/train')):
    di[index] = name
    
for name in os.listdir('../input/cnn-img-data/tiny-imagenet-200/test/images'):
    li.append(name)
li[3]
final= []

for p, name in zip(predList, li):
    final.append((name, di[p]))
len(final), len(li), final[0]
final.sort()
opt_fl = 'output.txt'
with open(opt_fl, 'w') as f:
    for name, p in final:
        f.write(' '.join([name, p, '\n']))
with open(opt_fl) as f:
    print(f.readline())
    print(f.readline())
    print(f.readline())
