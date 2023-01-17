%matplotlib inline 

# If we don't do this then image will open as pop-up and not in notebook
import pandas as pd

import numpy as np

from PIL import Image as im

#import matplotlib as plt

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import torch

from matplotlib.pyplot import imshow
#from skimage import io

#from skimage.viewer import ImageViewer
import wget

import time

import os

import copy
!ls -l ~/datasets/head-CT-hemorrhage/
labels = pd.read_csv("~/datasets/head-CT-hemorrhage/labels.csv")
labels.shape
labels
labels.columns
labels.dtypes
# PyTorch databuild libraries and modules

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, models, transforms
img1 = mpimg.imread('~/datasets/head-CT-hemorrhage/head_ct/000.png')

img1.shape
type(img1)
imgplot = plt.imshow(img1)
img2 = mpimg.imread('~/datasets/head-CT-hemorrhage/head_ct/010.png')

img2.shape
img1
img1.max(), img1.min()
tt = (img1 * 255).astype(np.uint8)

tt.max(), tt.min()
# So we force it from float to uint

tt = im.fromarray((img1 * 255).astype(np.uint8))

tt
type(tt)
class HeadHemorrhageDataset(Dataset):

    """CT scans of the head hemorrhage dataset."""

    

    def __init__(self, root_dir, label_file, transform=None):

        """

        Args:

            root_dir (string): root_dir point to the directory with image data.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.root_dir = root_dir

        self.labels = pd.read_csv(label_file)

        self.transform = transform



    def __len__(self):        

        return len(os.listdir(self.root_dir))



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

            

        image_id = idx

        if len(str(idx))==1:

            idx = '00'+str(idx)

        if len(str(idx))==2:

            idx = '0'+str(idx)

                    

        img_name = str(idx)+'.png'

        

        #print('Image id: '+str(image_id))

        # columns from class_map: image_id, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme

        #img_label = labels.loc[labels['id'] == image_id, ' hemorrhage'].to_numpy()

        img_label = self.labels.loc[image_id, ' hemorrhage']

        #print('Image label: '+str(img_label))

        # added to.numpy()[0] to remove index number

                

        img_path = os.path.join(self.root_dir,img_name)

        image = mpimg.imread(img_path)

        # This dataset contain a few 4-channel images towards the end. So ensure we select only first 3 channels as below

        image = image[:,:,:3]

        # convert from float to uint from 0 to 255

        tt=im.fromarray((image * 255).astype(np.uint8))

        

        if self.transform:

            img = self.transform(tt)

            # sample = {'img_label': img_label, 'image': img_data}



        return img, img_label
# Now let's create a PyTorch Dataset object with transformations

transformed_dataset = HeadHemorrhageDataset(root_dir='/home/ubuntu/datasets/head-CT-hemorrhage/head_ct/',

                                            label_file='/home/ubuntu/datasets/head-CT-hemorrhage/labels.csv',

                                           transform=transforms.Compose([

                                               transforms.Resize((224,224)),

                                               transforms.ToTensor(),

                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                                           ]))
tt = transformed_dataset.__getitem__(110)
len(tt), tt
tt[0].max(), tt[0].min()
# to_pil = transforms.ToPILImage() 

imshow(tt[0][0], cmap='gray')
tt = transformed_dataset.__getitem__(165)
len(tt), tt
for i in range(0,199):

    tt = transformed_dataset.__getitem__(i)

    print(i,tt[0].shape[0])
train_size = int(0.8 * len(transformed_dataset))

val_size = len(transformed_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(transformed_dataset, [train_size, val_size])
len(train_dataset), len(val_dataset)
model_data = {}

model_data['train'] = train_dataset

model_data['val'] = val_dataset
dataloaders = {x: DataLoader(model_data[x], 

                             batch_size=10,

                             #shuffle=True, 

                             num_workers=2)

              for x in ['train', 'val']}
len(dataloaders['train'].dataset), len(dataloaders['val'].dataset), len(dataloaders['train']), len(dataloaders['val'])
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
# class_names = dataloaders['train'].dataset.

dataset_sizes['train'], dataset_sizes['val']
len(dataloaders['train'].dataset), len(dataloaders['val'].dataset)
dataloaders['train'].dataset.indices
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            

            start_time = time.time()

            

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



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

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

            

            end_time = time.time()

            hours, rem = divmod(end_time-start_time, 3600)

            minutes, seconds = divmod(rem, 60)

            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
# PyTorch libraries and modules

from torch.optim import lr_scheduler

import torch.nn as nn

import torch.optim as optim

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load a pretrained model and reset final fully connected layer.



model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features



# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, 2)



# Ensuring the model is using GPU

model_ft = model_ft.to(device)



# As we have two classes (0 or 1) we will use cross-entropy as criterion

criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=15)
# Load a pretrained model and reset final fully connected layer.



model_ft = models.resnet152(pretrained=True)

num_ftrs = model_ft.fc.in_features



# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, 2)



# Ensuring the model is using GPU

model_ft = model_ft.to(device)



# As we have two classes (0 or 1) we will use cross-entropy as criterion

criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=10)