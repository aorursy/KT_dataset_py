# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir('/kaggle/input/nnfl-cnn-lab2/upload')

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import torch

from torch.utils.data import DataLoader, Dataset

import torch.nn as nn



import torchvision

from torchvision import transforms



import os

from PIL import Image
df = pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv')

df.head()
# Percentage disribution os labels

df.label.value_counts(normalize=True) * 100
class TrainImageDataset(Dataset):

    def __init__(self, csv_file, img_root, transform=None):

        self.dataframe = pd.read_csv(csv_file)

        self.img_root = img_root

        self.transform = transform

    

    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self, i):

        if torch.is_tensor(i):

            i = i.tolist()

        

        # Image

        img_name = os.path.join(self.img_root, self.dataframe.iloc[i, 0])

        image = Image.open(img_name)

        

        # Target

        target =  self.dataframe.iloc[i, 1]

        target = np.array([target]).reshape(-1)

        

        # Image Tranaforms

        if self.transform:

            image = self.transform(image)

            

        return image, target



class MapDataset(Dataset):

    def __init__(self, dataset, map_fn):

        self.dataset = dataset

        self.map = map_fn



    def __getitem__(self, index):

        image, label = self.dataset[index]

        return self.map(image), label



    def __len__(self):

        return len(self.dataset)
csv_file = '/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv'

img_root = '/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images'

IMAGE_SIZE = (150, 150)



data_transforms = {

    'train': transforms.Compose([

        transforms.Resize(IMAGE_SIZE),

        transforms.RandomRotation(20),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'val': transforms.Compose([

             transforms.Resize(IMAGE_SIZE),

             transforms.ToTensor(),

             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

             ])

}



train_dataset = TrainImageDataset(csv_file, img_root)

valid_size = int(len(train_dataset) * 0.2)

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-valid_size, valid_size])

train_dataset, valid_dataset = MapDataset(train_dataset, data_transforms['train']), MapDataset(valid_dataset, data_transforms['val'])



len(train_dataset), len(valid_dataset)
train_loader = DataLoader(train_dataset, batch_size=100, 

                          shuffle=True, num_workers=2)

valid_loader = DataLoader(valid_dataset, batch_size=32, 

                          shuffle=True, num_workers=2)
next(iter(train_loader))[1].shape
import matplotlib.pyplot as plt

import numpy as np



# functions to show an image

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    

# get some random training images

dataiter = iter(train_loader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images[:5]))

print(' '.join('%5s' % int(labels[j].numpy()) for j in range(5)))
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()

        

        # Conv Layer 1

        self.conv1 = nn.Conv2d(

            in_channels=in_channels, out_channels=out_channels,

            kernel_size=(3, 3), stride=stride, padding=1, bias=False

        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        

        # Conv Layer 2

        self.conv2 = nn.Conv2d(

            in_channels=out_channels, out_channels=out_channels,

            kernel_size=(3, 3), stride=1, padding=1, bias=False

        )

        self.bn2 = nn.BatchNorm2d(out_channels)

    

        # Shortcut connection to downsample residual

        # In case the output dimensions of the residual block is not the same 

        # as it's input, have a convolutional layer downsample the layer 

        # being bought forward by approporate striding and filters

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(

                    in_channels=in_channels, out_channels=out_channels,

                    kernel_size=(1, 1), stride=stride, bias=False

                ),

                nn.BatchNorm2d(out_channels)

            )



    def forward(self, x):

        out = nn.ReLU()(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = nn.ReLU()(out)

        return out

    

class ResNet(nn.Module):

    def __init__(self, num_classes=6):

        super(ResNet, self).__init__()

        

        # Initial input conv

        self.conv1 = nn.Conv2d(

            in_channels=3, out_channels=64, kernel_size=(3, 3),

            stride=1, padding=1, bias=False

        )



        self.bn1 = nn.BatchNorm2d(64)

        

        # Create blocks

        self.block1 = self._create_block(64, 64, stride=1)

        self.block2 = self._create_block(64, 128, stride=2)

        self.block3 = self._create_block(128, 256, stride=2)

        self.block4 = self._create_block(256, 512, stride=2)

        self.linear1 = nn.Linear(512 * 4 * 4, 512)

        self.linear2 = nn.Linear(512 , num_classes)

        self.relu = nn.ReLU()

    

    # A block is just two residual blocks for ResNet18

    def _create_block(self, in_channels, out_channels, stride):

        return nn.Sequential(

            ResidualBlock(in_channels, out_channels, stride),

            ResidualBlock(out_channels, out_channels, 1)

        )



    def forward(self, x):



        out = nn.ReLU()(self.bn1(self.conv1(x)))

        out = self.block1(out)

        out = self.block2(out)

        out = self.block3(out)

        out = self.block4(out)

        out = nn.AvgPool2d(4)(out)

        out = out.view(out.size(0), -1)

        out = self.linear1(out)

        out = self.relu(out)

        out = self.linear2(out)

        return out

    

model = ResNet()

x = torch.randn(1, 3, 150, 150)

model(x).shape
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = ResNet()

model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
# from tqdm.notebook import tqdm



# epochs = 50



# valid_min_loss = np.inf



# for epoch in range(epochs):

#     epoch_loss = []

#     epoch_accuracy = []

#     total = 0

#     model.train()

#     for image, label in tqdm(train_loader, total=len(train_loader)):

#         optimizer.zero_grad()

#         image, label = image.to(device).float(), label.to(device).long().view(-1)

#         output = model(image)

#         loss = criterion(output, label)

#         loss.backward()

#         optimizer.step()

#         epoch_loss.append(loss.item())

#         _, predicted = torch.max(output.data, 1)

#         correct = (predicted == label).float().sum()

#         accuracy = correct/label.size(0)

#         epoch_accuracy.append(accuracy)

#     epoch_accuracy = sum(epoch_accuracy)/len(epoch_accuracy)

#     epoch_loss = sum(epoch_loss)/len(epoch_loss)

    

#     valid_epoch_loss = []

#     valid_epoch_accuracy = []

#     total = 0

#     with torch.no_grad():

#         model.eval()

#         for image, label in valid_loader:

#             image, label = image.to(device).float(), label.to(device).long().view(-1)

#             output = model(image)

#             loss = criterion(output, label)

#             valid_epoch_loss.append(loss.item())

#             _, predicted = torch.max(output.data, 1)

#             correct = (predicted == label).float().sum()

#             accuracy = correct/label.size(0)

#             valid_epoch_accuracy.append(accuracy)

#         valid_epoch_accuracy = sum(valid_epoch_accuracy)/len(valid_epoch_accuracy)

#         valid_epoch_loss = sum(valid_epoch_loss)/len(valid_epoch_loss)

        

#     if valid_epoch_loss < valid_min_loss:

#         valid_min_loss = valid_epoch_loss

#         torch.save(model.state_dict(), 'model.h5')

#         print('Saving...')

#     print('Epoch : {:02d}\t Training Loss : {:.5f}\tValid Loss : {:.5f}\t Training Accuracy : {:.5f}\t Valid Accuracy : {:.5f}'.format(epoch+1, epoch_loss, valid_epoch_loss, epoch_accuracy, valid_epoch_accuracy))

        
# model = ResNet().to(device)

# model.load_state_dict(torch.load('model.h5'))

# model = model.eval()
# from tqdm.notebook import tqdm



# optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

# epochs = 30





# for epoch in range(epochs):

#     epoch_loss = []

#     epoch_accuracy = []

#     total = 0

#     model.train()

#     for image, label in tqdm(train_loader, total=len(train_loader)):

#         optimizer.zero_grad()

#         image, label = image.to(device).float(), label.to(device).long().view(-1)

#         output = model(image)

#         loss = criterion(output, label)

#         loss.backward()

#         optimizer.step()

#         epoch_loss.append(loss.item())

#         _, predicted = torch.max(output.data, 1)

#         correct = (predicted == label).float().sum()

#         accuracy = correct/label.size(0)

#         epoch_accuracy.append(accuracy)

#     epoch_accuracy = sum(epoch_accuracy)/len(epoch_accuracy)

#     epoch_loss = sum(epoch_loss)/len(epoch_loss)

    

#     valid_epoch_loss = []

#     valid_epoch_accuracy = []

#     total = 0

#     with torch.no_grad():

#         model.eval()

#         for image, label in valid_loader:

#             image, label = image.to(device).float(), label.to(device).long().view(-1)

#             output = model(image)

#             loss = criterion(output, label)

#             valid_epoch_loss.append(loss.item())

#             _, predicted = torch.max(output.data, 1)

#             correct = (predicted == label).float().sum()

#             accuracy = correct/label.size(0)

#             valid_epoch_accuracy.append(accuracy)

#         valid_epoch_accuracy = sum(valid_epoch_accuracy)/len(valid_epoch_accuracy)

#         valid_epoch_loss = sum(valid_epoch_loss)/len(valid_epoch_loss)

        

#     if valid_epoch_loss < valid_min_loss:

#         valid_min_loss = valid_epoch_loss

#         torch.save(model.state_dict(), 'model.h5')

#         print('Saving...')

#     print('Epoch : {:02d}\t Training Loss : {:.5f}\tValid Loss : {:.5f}\t Training Accuracy : {:.5f}\t Valid Accuracy : {:.5f}'.format(epoch+1, epoch_loss, valid_epoch_loss, epoch_accuracy, valid_epoch_accuracy))

        
# model = ResNet().to(device)

# model.load_state_dict(torch.load('model.h5'))

# model = model.eval()
# from tqdm.notebook import tqdm



# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

# epochs = 15





# for epoch in range(epochs):

#     epoch_loss = []

#     epoch_accuracy = []

#     total = 0

#     model.train()

#     for image, label in tqdm(train_loader, total=len(train_loader)):

#         optimizer.zero_grad()

#         image, label = image.to(device).float(), label.to(device).long().view(-1)

#         output = model(image)

#         loss = criterion(output, label)

#         loss.backward()

#         optimizer.step()

#         epoch_loss.append(loss.item())

#         _, predicted = torch.max(output.data, 1)

#         correct = (predicted == label).float().sum()

#         accuracy = correct/label.size(0)

#         epoch_accuracy.append(accuracy)

#     epoch_accuracy = sum(epoch_accuracy)/len(epoch_accuracy)

#     epoch_loss = sum(epoch_loss)/len(epoch_loss)

    

#     valid_epoch_loss = []

#     valid_epoch_accuracy = []

#     total = 0

#     with torch.no_grad():

#         model.eval()

#         for image, label in valid_loader:

#             image, label = image.to(device).float(), label.to(device).long().view(-1)

#             output = model(image)

#             loss = criterion(output, label)

#             valid_epoch_loss.append(loss.item())

#             _, predicted = torch.max(output.data, 1)

#             correct = (predicted == label).float().sum()

#             accuracy = correct/label.size(0)

#             valid_epoch_accuracy.append(accuracy)

#         valid_epoch_accuracy = sum(valid_epoch_accuracy)/len(valid_epoch_accuracy)

#         valid_epoch_loss = sum(valid_epoch_loss)/len(valid_epoch_loss)

        

#     if valid_epoch_loss < valid_min_loss:

#         valid_min_loss = valid_epoch_loss

#         torch.save(model.state_dict(), 'model.h5')

#         print('Saving...')

#     print('Epoch : {:02d}\t Training Loss : {:.5f}\tValid Loss : {:.5f}\t Training Accuracy : {:.5f}\t Valid Accuracy : {:.5f}'.format(epoch+1, epoch_loss, valid_epoch_loss, epoch_accuracy, valid_epoch_accuracy))

        
model = ResNet().to(device)

model.load_state_dict(torch.load('model.h5'))

model = model.eval()
model.to(device)

valid_epoch_accuracy = []

with torch.no_grad():

    model.eval()

    for image, label in valid_loader:

        image, label = image.to(device).float(), label.to(device).long().view(-1)

        output = model(image)

#         loss = criterion(output, label)

#         valid_epoch_loss.append(loss.item())

        _, predicted = torch.max(output.data, 1)

        correct = (predicted == label).float().sum()

        accuracy = correct/label.size(0)

        valid_epoch_accuracy.append(accuracy)

    valid_epoch_accuracy = sum(valid_epoch_accuracy)/len(valid_epoch_accuracy)

#     valid_epoch_loss = sum(valid_epoch_loss)/len(valid_epoch_loss)

print(valid_epoch_accuracy)
class TestImageDataset(Dataset):

    def __init__(self, img_root, transform=None):

        self.img_root = img_root

        self.transform = transform

        self.img_list = os.listdir(img_root)

    

    def __len__(self):

        return len(self.img_list)

    

    def __getitem__(self, i):

        if torch.is_tensor(i):

            i = i.tolist()

        

        # Image

        img_file = self.img_list[i]

        img_name = os.path.join(self.img_root, img_file)

        image = Image.open(img_name)

        idd = int(img_file.split('.')[0])

        

        # Image Tranaforms

        if self.transform:

            image = self.transform(image)

        return image, idd



class MapDataset(Dataset):

    def __init__(self, dataset, map_fn):

        self.dataset = dataset

        self.map = map_fn



    def __getitem__(self, index):

        image, idd = self.dataset[index]

        return self.map(image), idd



    def __len__(self):

        return len(self.dataset)

    

test_img_root = '/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images'

IMAGE_SIZE = (150, 150)



data_transforms_test = transforms.Compose([

             transforms.Resize(IMAGE_SIZE),

             transforms.ToTensor(),

             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

             ])



test_dataset = TestImageDataset(test_img_root)

test_dataset = MapDataset(test_dataset, data_transforms_test)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)



next(iter(test_loader))[0].shape
model.to(device)

test_dict = {}

with torch.no_grad():

    model.eval()

    for image, label in test_loader:

        image, label = image.to(device).float(), label.to(device).long().view(-1)

        output = model(image)

        _, predicted = torch.max(output.data, 1)

        label = int(label.cpu().numpy().reshape(-1))

        predicted = int(predicted.cpu().numpy().reshape(-1))

        test_dict[label] = predicted

test_dict
ids = []

labels = []

for id, label in test_dict.items():

    ids.append(id)

    labels.append(label)



test_csv = '/kaggle/input/nnfl-cnn-lab2/upload/sample_submission.csv'

submission = pd.read_csv(test_csv)



submission['image_name'] = ids

submission['label'] = labels

submission = submission.sort_values('image_name')

submission['image_name'] = submission['image_name'].astype(str) + '.jpg'

submission.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)
from IPython.display import FileLink

FileLink(r'model.h5')