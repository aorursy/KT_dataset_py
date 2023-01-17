# All the required imports



import pandas as pd

import math

import numpy as np

import os

import torch

import torchvision

from torchvision import transforms,models

from torch.autograd import Variable

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from torch import nn

import torch.nn.functional as F

from torch import optim

from skimage import io, transform

from torch.utils.data.sampler import SubsetRandomSampler





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
batch_size = 16

validation_split = .1

shuffle_dataset = True

random_seed= 42



# Transforms to be applied to each image (you can add more transforms), resizing every image to 3 x 224 x 224 size and converting to Tensor

transform = transforms.Compose([

                                transforms.RandomResizedCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                     std=[0.229, 0.224, 0.225])

                                ])



trainset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform) 

dataset_size = len(trainset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :

    np.random.seed(random_seed)

    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]#Training Dataset

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 

                                           sampler=train_sampler,num_workers = 0)

validationloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,

                                                sampler=valid_sampler,num_workers=0)

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
model = models.resnet152(pretrained = True)

model
model.classifier = nn.Sequential(

           nn.Linear(1000,512),

           nn.ReLU(),

           nn.Dropout(0.05),

          nn.Linear(512,128),

           nn.ReLU(),

           nn.Linear(128,67),

           nn.Softmax(dim=1))

model.cuda()
for name, child in model.named_children():

   if name in ['fc']:

       #print(name + ' is unfrozen')

       for param in child.parameters():

           param.requires_grad = True

   else:

       #print(name + ' is frozen')

       for param in child.parameters():

           param.requires_grad = True
optimizer = optim.Adadelta(model.parameters(),lr = 0.0001)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma=0.1, last_epoch=-1)

criterion = nn.CrossEntropyLoss()
batch_size = 16

n_iters = 10000

iter = 0

num_epochs = n_iters / (len(trainset) / batch_size)

num_epochs = 32

for epochs in range(num_epochs):

    for i,(images,labels) in enumerate(trainloader):

        images = Variable(images.cuda())

        labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        scheduler.step()

       

       

        iter += 1

        if iter%200 == 0:

                print('loss {}'.format(loss.data.item()))

            
model.eval()
correct = 0

total = 0

accuracy = 0

for images, labels in validationloader:

    # move tensors to GPU if CUDA is available

    

    

    images = images.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(images)

    _, predicted = torch.max(output, 1)

    total += labels.size(0)

    correct += (predicted.cpu() == labels.cpu()).sum()

            

    accuracy = 100 * correct / total

print(accuracy)

print(total)

print(accuracy)

   
model.eval()

submission = pd.read_csv('../input/sample_sub.csv')

submission.head()


testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
predictions = []

# iterate over test data to make predictions

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        



submission['Category'] = predictions             #Attaching predictions to submission file

submission.to_csv('submission.csv', index=False, encoding='utf-8')
import os

os.environ['KAGGLE_USERNAME'] = "perseusdg"

os.environ['KAGGLE_KEY'] = "e6d3aa705c08097cb10355c7e4663487"
!pip install kaggle

!kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "Message"