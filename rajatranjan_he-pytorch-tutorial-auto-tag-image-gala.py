# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from skimage import io, transform



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import os

from PIL import Image

from IPython.display import display



# Filter harmless warnings

import warnings

warnings.filterwarnings("ignore")
# TEST YOUR VERSION OF PILLOW

# Run this cell. If you see a picture of a cat you're all set!

with Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/image8247.jpg') as im:

    display(im)
dftrain=pd.read_csv('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/train.csv')

dftest=pd.read_csv('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/test.csv')

target_map={'Food':0, 'misc':1, 'Attire':2, 'Decorationandsignage':3}

# train

dftrain['Class'].unique()

dftrain['Class']=dftrain['Class'].map(target_map)
dftrain[:5000]
train=dftrain[:5000]

test=dftrain[5000:]
train['Class'].value_counts()
train.head()
# Start by creating a list

img_sizes = []

rejected = []



for item in train.Image:

    try:

        with Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/'+item) as img:

            img_sizes.append(img.size)

    except:

        rejected.append(item)

        

print(f'Images:  {len(img_sizes)}')

print(f'Rejects: {len(rejected)}')
df_img = pd.DataFrame(img_sizes)



# Run summary statistics on image widths

df_img[0].describe(),df_img[1].describe()
img1=Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/image9233.jpg')

print(img1.size)

display(img1)
r, g, b = img1.getpixel((0, 0))

print(r,g,b)
transform = transforms.Compose([

    transforms.ToTensor()

])

im = transform(img1)

print(im.shape)

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
im[:,0,0]
transform = transforms.Compose([

    transforms.Resize(224), 

    transforms.ToTensor()

])

im = transform(img1)

print(im.shape)

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=1),  # normally we'd set p=0.5

    transforms.RandomRotation(30),

    transforms.Resize(224),

    transforms.CenterCrop(224), 

    transforms.ToTensor()

])

im = transform(img1)

print(im.shape)

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],

                         [0.229, 0.224, 0.225])

])

im = transform(img1)

print(im.shape)

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
# After normalization:

im[:,0,0]
inv_normalize = transforms.Normalize(

    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],

    std=[1/0.229, 1/0.224, 1/0.225]

)

im_inv = inv_normalize(im)

plt.figure(figsize=(12,4))

plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
plt.figure(figsize=(12,4))

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
train_transform = transforms.Compose([

        transforms.RandomRotation(10),      # rotate +/- 10 degrees

        transforms.RandomHorizontalFlip(),  # reverse 50% of images

        transforms.Resize(224),             # resize shortest side to 224 pixels

        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ])



test_transform = transforms.Compose([

        transforms.Resize(224),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ])
class CustomDataset(Dataset):



    def __init__(self, imgz,labels=None, root_dir='', transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.images = imgz

        self.labels = labels

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        img_name = os.path.join(self.root_dir,self.images[idx])

        image = Image.open(img_name).convert('RGB')

#         plt.imshow(image)

        if self.transform:

            image = self.transform(image)

        if self.labels is not None:

            return image, self.labels[idx]

        else:

            return image
train_data = CustomDataset(imgz=train['Image'].values,labels=train['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/',transform=train_transform)

test_data = CustomDataset(imgz=test['Image'].values,labels=test['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/', transform=test_transform)

total_test_data = CustomDataset(imgz=dftest['Image'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Test Images/', transform=test_transform)


torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

total_test_loader = DataLoader(total_test_data, batch_size=4, shuffle=False)



class_names = train.Class.unique()



print(class_names)

print(f'Training images available: {len(train_data)}')

print(f'Testing images available:  {len(test_data)}')
total_train_data = CustomDataset(imgz=dftrain['Image'].values,labels=dftrain['Class'].values,root_dir='/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Train Images/',transform=train_transform)

total_train_loader = DataLoader(total_train_data, batch_size=4, shuffle=True)
train_data.__getitem__(2644)
# Grab the first batch of 10 images



target_map_inv={0:'Food', 1:'misc', 2:'Attire', 3:'Decorationandsignage'}

from torchvision.utils import make_grid

for images,labels in train_loader: 

    break



# Print the labels

print('Label:', labels.numpy())



print('Class:', *np.array([target_map_inv[i.tolist()] for i in labels]))



im = make_grid(images,nrow=8)  # the default nrow is 8



# Inverse normalize the images

inv_normalize = transforms.Normalize(

    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],

    std=[1/0.229, 1/0.224, 1/0.225]

)

im_inv = inv_normalize(im)



# Print the images

plt.figure(figsize=(12,4))

plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
class ConvolutionalNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3, 1)

        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        self.fc1 = nn.Linear(54*54*16, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 4)



    def forward(self, X):

        X = F.relu(self.conv1(X))

        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.conv2(X))

        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, 54*54*16)

        X = F.relu(self.fc1(X))

        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

torch.manual_seed(101)

CNNmodel = ConvolutionalNetwork()

criterion = nn.CrossEntropyLoss()



if torch.cuda.is_available():

    CNNmodel = CNNmodel.cuda()

    criterion = criterion.cuda()

    

    

# optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)

optimizer = torch.optim.SGD(CNNmodel.parameters(), lr=0.001, momentum=0.9)

CNNmodel
def count_parameters(model):

    params = [p.numel() for p in model.parameters() if p.requires_grad]

    for item in params:

        print(f'{item:>8}')

    print(f'________\n{sum(params):>8}')
count_parameters(CNNmodel)
print(train.shape,test.shape)


def test_model(net, testloader):

    correct = 0

    total = 0

    with torch.no_grad():

        for b, (X_test, y_test) in enumerate(test_loader):

            if torch.cuda.is_available():

                X_test = X_test.cuda()

                y_test = y_test.cuda()

            

            inputs, labels = X_test, y_test

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

 

    print('Accuracy of the network on test images: %0.3f %%' % (100 * correct / total))

def train_model(net, trainloader,epochs):

    for epoch in range(epochs): # no. of epochs

        running_loss = 0

#         exp_lr_scheduler.step()

        for b, (X_train, y_train) in enumerate(train_loader):

            # data pixels and labels to GPU if available

            b+=1

            

            if torch.cuda.is_available():

                X_train = X_train.cuda()

                y_train = y_train.cuda()

            inputs, labels = X_train, y_train

            # set the parameter gradients to zero

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            # propagate the loss backward

            loss.backward()

            # update the gradients

            optimizer.step()

 

            running_loss += loss.item()

        print('[Epoch %d] loss: %.3f' %

                      (epoch + 1, running_loss/len(trainloader)))

        

 

    print('Done Training')

    

# plt.plot(train_losses, label='training loss')

# plt.plot(test_losses, label='validation loss')

# plt.title('Loss at the end of each epoch')

# plt.legend();
# plt.plot([t for t in train_correct], label='training accuracy')

# plt.plot([t for t in test_correct], label='validation accuracy')

# plt.title('Accuracy at the end of each epoch')

# plt.legend();
# print(test_correct)

# print(f'Test accuracy: {test_correct[-1].item()*100/3000:.3f}%')
from torchvision import datasets, transforms, models # add models to the list

AlexNetmodel = models.alexnet(pretrained=True)

AlexNetmodel
for param in AlexNetmodel.parameters():

    param.requires_grad = False
torch.manual_seed(42)

# AlexNetmodel.fc = nn.Sequential(nn.Linear(9216, 1024),

AlexNetmodel.classifier = nn.Sequential(nn.Linear(9216, 1024),

                                 nn.ReLU(),

                                 nn.Dropout(0.4),

                                 nn.Linear(1024, 4),

                                 nn.LogSoftmax(dim=1))



# for param in AlexNetmodel.fc.parameters():

#     param.requires_grad = True

AlexNetmodel
# These are the TRAINABLE parameters:

count_parameters(AlexNetmodel)
from torch.optim import lr_scheduler





criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():

    AlexNetmodel = AlexNetmodel.cuda()

    criterion = criterion.cuda()

    

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    

optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr=0.0001)

# optimizer = torch.optim.SGD(AlexNetmodel.classifier.parameters(), lr=0.0001, momentum=0.9)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
torch.cuda.is_available()
train_model(AlexNetmodel,train_loader,5)

test_model(AlexNetmodel,test_loader)
def prediciton(net, data_loader):

    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):

        if torch.cuda.is_available():

            data = data.cuda()

        output = net(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

    

    return test_pred







train_model(AlexNetmodel,total_train_loader,15)



test_pred = prediciton(AlexNetmodel, total_test_loader)
test_pred.numpy().shape
dftest['Class']=test_pred.numpy()

dftest.Class=dftest.Class.map(target_map_inv)
dftest.to_csv('s6_alexnet.csv',index=False)
dftest
img1=Image.open('/kaggle/input/hackerearth-dl-challengeautotag-images-of-gala/dataset/Test Images/image3442.jpg')

print(img1.size)

display(img1)