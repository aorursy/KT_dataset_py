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



from PIL import Image



%matplotlib inline 
# Exploring train.csv file

df = pd.read_csv('../input/train.csv')

df.head()


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
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([transforms.Resize(256),

                                transforms.CenterCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                ])

batch_size = 16

validation_split = .35

shuffle_dataset = True

random_seed= 40



dataset = ImageDataset(csv_file = '../input/train.csv', root_dir = '../input/data/data/', transform=transform)



# Creating data indices for training and validation splits:

dataset_size = len(dataset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :

    np.random.seed(random_seed)

    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)

dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 

                                           sampler=train_sampler)

dataloaders['valid'] = torch.utils.data.DataLoader(dataset, batch_size=batch_size,

                                                sampler=valid_sampler)



dataset_sizes = {

    'train': dataset_size * (1-validation_split),

    'valid': dataset_size * validation_split

}
# check if CUDA / GPU is available, if unavaiable then turn it on from the right side panel under SETTINGS, also turn on the Internet

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# Training Loop (You can write your own loop from scratch)

import time

import os

import copy

def train_model(model, criterion, optimizer, num_epochs=15, early_stop_epoch=4):

    since = time.time()

    no_acc_change_epoch = 0

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{} -->'.format(epoch, num_epochs - 1))



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if (phase == 'train'):

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                if train_on_gpu:

                    inputs = inputs.cuda()

                    labels = labels.cuda()



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid':

                if epoch_acc > best_acc:

                    best_acc = epoch_acc

                    best_model_wts = copy.deepcopy(model.state_dict())

                    no_acc_change_epoch = 0

                else:

                    no_acc_change_epoch += 1

        if no_acc_change_epoch >= early_stop_epoch:

            print("Early Stopping")

            break

        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best validation Acc: {:4f}'.format(best_acc))

    

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
from torchvision import models
res_model = models.vgg19(pretrained=True)

for param in res_model.parameters():

    param.requires_grad = False

res_model.classifier
res_model.classifier[6] = nn.Sequential(

                            nn.Linear(4096, 2048),

                            nn.ReLU(),

                            nn.Dropout(p=0.5),

                            nn.Linear(2048, 67),

                            nn.LogSoftmax(dim=1))

res_model.classifier
if train_on_gpu:

    res_model.cuda()

# Loss function to be used

criterion = nn.CrossEntropyLoss()   # You can change this if needed



# Optimizer to be used, replace "your_model" with the name of your model and enter your learning rate in lr=0.001

optimizer = optim.Adagrad(res_model.parameters(), lr=0.01)

print(criterion)

print(optimizer)
best_model = train_model(res_model, criterion, optimizer)
submission = pd.read_csv('../input/sample_sub.csv')

submission.head()
transform_test = transforms.Compose([transforms.Resize(size=256),

                                     transforms.CenterCrop(size=224),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset = ImageDataset(csv_file = '../input/sample_sub.csv', root_dir = '../input/data/data/', transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)
predictions = []

for data, target in testloader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = best_model(data)

    _, pred = torch.max(output, 1)

    for i in range(len(pred)):

        predictions.append(int(pred[i]))

        

submission['Category'] = predictions             #Attaching predictions to submission file
submission.to_csv('submission.csv', index=False, encoding='utf-8')
!pip install kaggle
import os

os.environ['KAGGLE_USERNAME'] = 'nousernameforme'

os.environ['KAGGLE_KEY'] = ''
#!kaggle competitions submit -c qstp-deep-learning-2019 -f submission.csv -m "VGG19 L(4096, 2048 , 67) Relu Dropout 0.5 Adagrad 0.01"