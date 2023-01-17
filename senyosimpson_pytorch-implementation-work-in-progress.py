import logging

import os

import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import transforms

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from PIL import Image
print(os.listdir("../input"))
class CapeTownTransportation(Dataset):

    def __init__(self, csv, train_dir, transform=None):

        """

        Args:

            csv (str) : path to a csv file containing image paths and labels

            train_dir (str): the root directory containing the training images

            transform (torch.Compose): transforms to apply to images

        """

        self.dataset = pd.read_csv(csv)

        self.train_dir = train_dir

        self.transform = transform

        self.classes = {'city_sightseeing_bus' : 0,

                        'train' : 1,

                        'golden_arrow_bus' : 2,

                        'myciti_bus' : 3,

                        'uct_jammie_shuttle' : 4,

                        'minibus_taxi' : 5

                       }



    def __len__(self):

        return len(self.dataset)



    def __getitem__(self, idx):

        image_id = self.dataset['id'][idx]

        path_to_image = os.path.join(self.train_dir, image_id)

        image = Image.open(path_to_image).convert("RGB")

        label_name = self.dataset['category'][idx]

        label_value = self.encode_label(label_name)

        label_value = np.array([label_value]) # necessary for matching dimensions

        label_value = torch.LongTensor(label_value)

        if self.transform:

            image = self.transform(image)

    

        sample = (image, label_value)

        return sample

            

    def encode_label(self, label_name):

        """Assigns the label value to the label

        name of a class"""

        return self.classes[label_name]
# TODO - Add batchnorm



class ResidualUnit(nn.Module):

    """Residual unit for the ResNet architecture"""

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 

                               kernel_size=3, stride=stride, 

                               padding=1, bias=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 

                               kernel_size=3, stride=1,

                               padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = shortcut



    def forward(self, x):

        identity = x

        x = self.relu(self.conv1(x))

        x = self.conv2(x)

        if self.shortcut:

            identity = self.shortcut(identity)

        x = torch.add(x, identity)

        x = self.relu(x)

        return x



class ResNet18(nn.Module):

    """Implementation of ResNet18 architecture"""

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.unit1 = self._make_unit(strides=[1, 1], in_channels=64, out_channels=64)

        self.unit2 = self._make_unit(strides=[2, 1], in_channels=64, out_channels=128)

        self.unit3 = self._make_unit(strides=[2, 1], in_channels=128, out_channels=256)

        self.unit4 = self._make_unit(strides=[2, 1], in_channels=256, out_channels=512)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.affine = nn.Linear((512*7*7), 10)



    def _make_unit(self, strides, in_channels, out_channels):

        layers = []

        if strides[0] != 1 or in_channels != out_channels:

            shortcut = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides[0], bias=False)

            )

            layers.append(ResidualUnit(in_channels, out_channels, strides[0], shortcut))



        for _ in range(1, len(strides)):

            layers.append(ResidualUnit(out_channels, out_channels))

        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv(x)

        x = self.maxpool(x)

        x = self.unit1(x)

        x = self.unit2(x)

        x = self.unit3(x)

        x = self.unit4(x)

        x = self.avgpool(x)

        x = x.view(-1, 512*7*7)

        out = self.affine(x)

        return out
def train(logger):

    """

    Train neural network

    Args:

        logger(<>) : logger object to log results

    """

    # set hyperparameters

    model_name = 'ResNet18'

    batch_size = 8

    n_epochs = 1

    optimizer = optim.SGD

    lr = 1e-2

    momentum = 0.9

    use_nesterov = True

    

    model = ResNet18()

    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, nesterov=use_nesterov)

    objective_fn = nn.CrossEntropyLoss()

    

    transform = transforms.Compose([transforms.Resize((224,224)),

                                    transforms.RandomHorizontalFlip(),

                                    transforms.RandomVerticalFlip(),

                                    transforms.RandomRotation(180),

                                    transforms.ToTensor()

                                   ])

    

    # get dataset and create dataloader

    ds = CapeTownTransportation(csv='../input/labels.csv',

                                train_dir='../input/train',

                                transform=transform)

    dataloader = DataLoader(ds, batch_size=batch_size,

                           shuffle=True) #num_workers=2)

    

    for epoch in range(n_epochs):

        logger.info(' ------------ Epoch {}/{} -------------- '.format(epoch+1, n_epochs))

        for step, batch in enumerate(dataloader):

            images, labels = batch

            labels = labels.view(-1)

            optimizer.zero_grad()

            outputs = model(images)

            loss = objective_fn(outputs, labels)

            loss.backward()

            optimizer.step()



            outputs = torch.argmax(outputs, dim=1)

            accuracy = ((outputs.detach().numpy() == labels.detach().numpy()).sum()) / batch_size



            logger.info('Step {}, Loss {}, Accuracy {}%'.format(step+1,

                                                                 round(loss.item(), 3),

                                                                 round(accuracy*100, 3)))
# setup logging

logger = logging.getLogger('results')

logger.setLevel(logging.INFO)

ch = logging.StreamHandler()

ch.setLevel(logging.INFO)

logger.addHandler(ch)



# running training job

train(logger)