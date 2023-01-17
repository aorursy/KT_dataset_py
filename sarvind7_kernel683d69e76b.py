import torch

import torchvision

import torchvision.transforms as transforms

transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/gnr638-mls4rs-a1/gnr638_train/train'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

# for dirname, _, filenames in os.walk('/kaggle/input/facess'):

#     print (dirname,_,filenames)



import pandas as pd

landmarks_frame = pd.read_csv('/kaggle/input/facess/faces/face_landmarks.csv')  

img_name = landmarks_frame.iloc[65,0]

data_points=landmarks_frame.iloc[65,1:].values

# len(data_points)

landmarks = data_points.astype('float').reshape(-1, 2)
from __future__ import print_function, division

import os

import torch

import pandas as pd

from skimage import io, transform

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



plt.ion()   # interactive mode
def show_landmarks(image, landmarks):

    """Show image with landmarks"""

    plt.imshow(image)

    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')

    plt.pause(0.001)  # pause a bit so that plots are updated



plt.figure()

show_landmarks(io.imread(os.path.join('/kaggle/input/facess/faces/', img_name)),

               landmarks)

plt.show()
import torch

import torchvision

import torchvision.transforms as transforms

transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)
class FaceLandmarksDataset(Dataset):

    """Face Landmarks dataset."""



    def __init__(self,root_dir='/kaggle/input/gnr638-mls4rs-a1/gnr638_train/train', transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        

#         self.landmarks_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        self.train_label=[]

        self.train_img=[]

        for dirname, _, filenames in os.walk(root_dir):

            for filename in filenames:

                self.train_label.append(dirname.split('/')[-1])

                self.train_img.append(os.path.join(dirname, filename))

        self.all_data=list(zip(self.train_label,self.train_img))



    def __len__(self):

        return len(self.all_data)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        img_name = self.all_data[idx][1]

        image = io.imread(img_name)

        sample = (image,self.all_data[idx][0])



        if self.transform:

            sample = self.transform(sample)



        return sample

img_source=FaceLandmarksDataset()

import numpy as np

x=np.array(img_source[2][0])

x=np.resize(x,(256,256))

x.shape
class Rescale(object):

    """Rescale the image in a sample to a given size.



    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image= sample[0]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), preserve_range=True).astype(int)

        # h and w are swapped for landmarks because for images,

        # x and y axes are axis 1 and 0 respectively

#         print(img)

        return (img,sample[1])





class RandomCrop(object):

    """Crop randomly the image in a sample.



    Args:

        output_size (tuple or int): Desired output size. If int, square crop

            is made.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, sample):

        image= sample[0]

        h, w = image.shape[:2]

        new_h, new_w = self.output_size



        top = np.random.randint(0, h - new_h)

        left = np.random.randint(0, w - new_w)



        image = image[top: top + new_h,

                      left: left + new_w]



        

#         print(image)

        return (image,sample[1])





class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image= sample[0]

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        #print(image)

        return (torch.from_numpy(image),sample[1])

transformed_dataset = FaceLandmarksDataset(

                                           transform=transforms.Compose([

                                               Rescale(40),

                                               RandomCrop(32),

                                               ToTensor()

                                           ]))



trainloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=4,

                                          shuffle=True, num_workers=2)

x=transformed_dataset[0]

a,b=x

a
a.shape
dataiter = iter(trainloader)

dir(dataiter)

images, labels = dataiter.__next__()

# print (images.shape)

plt.imshow(images)

# print(images.shape,labels)
import torch

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np



# functions to show an image





def imshow(img):

#     img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

#     print (img)

    plt.imshow(img)

    plt.show()

imshow(torchvision.utils.make_grid(images))

classes = ('aircrafts','ships','none') 
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 3)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        labels=torch.tensor(labels)

        print (type(inputs),type(labels))

        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs.float())

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 200 == 199:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 200))

            running_loss = 0.0



print('Finished Training')