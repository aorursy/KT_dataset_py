# All the required imports



import pandas as pd

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torchvision import models

from PIL import Image

%matplotlib inline 
#Dataset class



class ImageDataset(Dataset):

    



    def __init__(self, csv_file, root_dir, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.data_frame = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data_frame)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data_frame['Id'][idx])         # getting path of image

        image = Image.open(img_name).convert('RGB')                                # reading image and converting to rgb if it is grayscale

        label = np.array(self.data_frame['Category'][idx])                         # reading label of the image

        

        if self.transform:            

            image = self.transform(image)                                          # applying transforms, if any

        

        sample = (image, label)        

        return sample
# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.ToTensor()                               

                                ])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomRotation(15),

                                transforms.ToTensor()                               

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data', transform=train_transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
model=models.densenet161(pretrained=True)

model.classifier=nn.Linear(2208,67,bias=True)
model

##### criterion = nn.CrossEntropyLoss()   # You can change this if needed

# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay=1e-5)

model = model.to('cuda')
#epochs were 22 or 24

n_epochs=5

model.train()

#number of epochs, change this accordingly

for epoch in range(1, n_epochs+1):

    LOSS=0

    total_train=0

    correct_train=0

    for inputs, labels in trainloader:

         # Move input and label tensors to the default device

        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        logps = model(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()

        LOSS=LOSS+loss.item()

        _, predicted = torch.max(logps.data, 1)

        total_train += labels.nelement()

        correct_train += predicted.eq(labels.data).sum().item()

        train_accuracy = 100 * correct_train / total_train

    print(train_accuracy)

    print(LOSS/len(trainloader))

    print(epoch)
model.eval() # eval mode

# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

#Loading test data to make predictions

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

predictions=[]

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

submission['Category'] = predictions             #Attaching predictions to submission file

#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')

os.environ['KAGGLE_USERNAME']="someshsingh22"

os.environ['KAGGLE_KEY']="f0df941af22f7d0691f641ada69e8154"

!pip install kaggle

!kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "4d"
model.eval() # eval mode

# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')
#Loading test data to make predictions



testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)

predictions=[]

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

submission['Category'] = predictions             #Attaching predictions to submission file

#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')