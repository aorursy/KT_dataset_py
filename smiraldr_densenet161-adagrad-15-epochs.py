# All the required imports



import pandas as pd

import numpy as np

from torchvision import transforms, datasets, models

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

import os

import numpy as np

import pandas as pd

import torch

import torchvision

import torch.nn as nn

import cv2

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms, utils

import copy

import tqdm

import sys



from PIL import Image



%matplotlib inline 
# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()
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



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)     #Training Dataset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)                     #Train loader, can change the batch_size to your own choice
#Checking training sample size and label

for i in range(len(trainset)):

    sample = trainset[i]

    print(i, sample[0].size(), " | Label: ", sample[1])

    if i == 9:

        break
# Visualizing some sample data

# obtain one batch of training images

dataiter = iter(trainloader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):                                             #Change the range according to your batch-size

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
model = models.densenet161(pretrained=True)



device='cuda'

for param in model.parameters():

    param.requires_grad = False

num_ftrs = model.classifier.in_features

model.classifier = nn.Linear(num_ftrs, 67)



model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.005)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30000,60000,90000], gamma=0.3)
child_counter = 0

for child in model.children():

    print(" child", child_counter, "is:")

    print(child)

    child_counter += 1
epochs = 8

itr = 1

p_itr = 5000

total_loss, score = 0,0

val_score = 0.0

best_model_wts = copy.deepcopy(model.state_dict())

best_score = 0.0

loss_list = []

score_list = []





for epoch in range(epochs):

    model.train()

    for samples, labels in trainloader:

        samples, labels = samples.to(device), labels.to(device)    

        optimizer.zero_grad()

        output = model(samples)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        scheduler.step()
# Reading sample_submission file to get the test image names

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
#Loading test data to make predictions

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

predictions=[]
model.eval()

# iterate over test data to make predictions

for images, labels in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        images, labels = images.cuda(), labels.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(images)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



submission['Category'] = predictions             #Attaching predictions to submission file

#saving submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')
import os

os.environ['KAGGLE_USERNAME']='smiraldr'

os.environ['KAGGLE_KEY']='af95878e7e56a523ae5757c5ebdb85c3'
!pip install kaggle --upgrade
!kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "densenet161 0.005 unnormalized"