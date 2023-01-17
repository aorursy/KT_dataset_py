# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import lr_scheduler

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

from torchvision import models

import torch.optim as optim

import pandas as pd

import numpy as np

import cv2

import os

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline

import time

from tqdm import tqdm
from pathlib import Path

import urllib



import os

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
input_root_dir = "../input/food-101/food-101/food-101"

input_root_path = Path(input_root_dir)

print(os.listdir(input_root_dir))

image_dir_path = input_root_path/'images'
!cat {input_root_dir}/README.txt
class_path = input_root_dir+'/meta/classes.txt'

train_img_name_path = input_root_dir+'/meta/train.txt'

test_img_name_path = input_root_dir+'/meta/test.txt'

def file2list(path):

    file1 = open(path,'r')

    lines = file1.readlines()

    final_list = [line.strip() for line in lines]

    return final_list
classes = file2list(class_path)

train_data = file2list(train_img_name_path)

test_data = file2list(test_img_name_path)

le = preprocessing.LabelEncoder()

targets = le.fit_transform(classes)
class FoodData(Dataset):

    def __init__(self,img_path,img_dir,size,transform=None):

        self.img_path = img_path

        self.img_dir = img_dir

        self.transform = transform

        self.size = size

#         self.mode = mode

        

    def __len__(self):

        return len(self.img_path)

    

    def __getitem__(self,index):

        label,img_name = self.img_path[index].split('/')

        path = self.img_dir+'/images/'+label+'/'+img_name+'.jpg'

        img = cv2.imread(path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(self.size,self.size))

        if self.transform:

            img = self.transform(img)

        return {

                'gt': img,

                'label': torch.tensor(le.transform([label])[0])

            }

        
class Cutout(object):

    """Randomly mask out one or more patches from an image.

    Args:

        n_holes (int): Number of patches to cut out of each image.

        length (int): The length (in pixels) of each square patch.

    """

    def __init__(self, n_holes, length):

        self.n_holes = n_holes

        self.length = length



    def __call__(self, img):

        """

        Args:

            img (Tensor): Tensor image of size (C, H, W).

        Returns:

            Tensor: Image with n_holes of dimension length x length cut out of it.

        """

        h = img.size(1)

        w = img.size(2)



        mask = np.ones((h, w), np.float32)



        for n in range(self.n_holes):

            y = np.random.randint(h)

            x = np.random.randint(w)



            y1 = np.clip(y - self.length // 2, 0, h)

            y2 = np.clip(y + self.length // 2, 0, h)

            x1 = np.clip(x - self.length // 2, 0, w)

            x2 = np.clip(x + self.length // 2, 0, w)



            mask[y1: y2, x1: x2] = 0.



        mask = torch.from_numpy(mask)

        mask = mask.expand_as(img)

        img = img * mask



        return img
from PIL import Image, ImageEnhance, ImageOps

import numpy as np

import random





class ImageNetPolicy(object):

    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:

        >>> policy = ImageNetPolicy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     ImageNetPolicy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),

            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),



            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),

            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),

            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),

            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),

            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),



            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),

            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),

            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),

            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),

            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),



            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),

            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),

            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),

            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),

            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),



            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),

            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),

            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment ImageNet Policy"





class CIFAR10Policy(object):

    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:

        >>> policy = CIFAR10Policy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     CIFAR10Policy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),

            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),

            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),

            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),



            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),

            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),

            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),

            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),

            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),



            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),

            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),

            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),

            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),

            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),



            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),

            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),

            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),

            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),

            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),



            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),

            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),

            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment CIFAR10 Policy"





class SVHNPolicy(object):

    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:

        >>> policy = SVHNPolicy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     SVHNPolicy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),

            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),

            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),

            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),



            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),

            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),

            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),

            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),

            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),



            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),

            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),

            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),

            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),

            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),



            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),

            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),

            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),

            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),



            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),

            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),

            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),

            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),

            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment SVHN Policy"





class SubPolicy(object):

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):

        ranges = {

            "shearX": np.linspace(0, 0.3, 10),

            "shearY": np.linspace(0, 0.3, 10),

            "translateX": np.linspace(0, 150 / 331, 10),

            "translateY": np.linspace(0, 150 / 331, 10),

            "rotate": np.linspace(0, 30, 10),

            "color": np.linspace(0.0, 0.9, 10),

            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),

            "solarize": np.linspace(256, 0, 10),

            "contrast": np.linspace(0.0, 0.9, 10),

            "sharpness": np.linspace(0.0, 0.9, 10),

            "brightness": np.linspace(0.0, 0.9, 10),

            "autocontrast": [0] * 10,

            "equalize": [0] * 10,

            "invert": [0] * 10

        }



        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand

        def rotate_with_fill(img, magnitude):

            rot = img.convert("RGBA").rotate(magnitude)

            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)



        func = {

            "shearX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "shearY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "translateX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),

                fillcolor=fillcolor),

            "translateY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),

                fillcolor=fillcolor),

            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),

            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),

            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),

            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),

            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),

            "equalize": lambda img, magnitude: ImageOps.equalize(img),

            "invert": lambda img, magnitude: ImageOps.invert(img)

        }



        self.p1 = p1

        self.operation1 = func[operation1]

        self.magnitude1 = ranges[operation1][magnitude_idx1]

        self.p2 = p2

        self.operation2 = func[operation2]

        self.magnitude2 = ranges[operation2][magnitude_idx2]





    def __call__(self, img):

        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)

        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)

        return img
# from autoaugment import ImageNetPolicy

from albumentations.pytorch import ToTensor

import albumentations as A

augmentation_pipeline = A.Compose(

    [

        A.HorizontalFlip(p = 0.5), # apply horizontal flip to 50% of images

        A.VerticalFlip(p =0.5),

        A.OneOf(

            [

                # apply one of transforms to 50% of images

                A.RandomContrast(), # apply random contrast

                A.RandomGamma(), # apply random gamma

                A.RandomBrightness(), # apply random brightness

                A.RandomBrightnessContrast(),

            ],

            p = 0.5

        ),

        A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=0.5),

        A.Blur(blur_limit=(15, 15), p=0.5),

        A.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]),

        

        ToTensor() # convert the image to PyTorch tensor

    ],

    p = 1

)

transforms_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomRotation(90),

#     transforms.CenterCrop(10),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ColorJitter(),

#     ImageNetPolicy(),

    

    transforms.ToTensor(),

    transforms.Normalize( mean = np.array([0.485, 0.456, 0.406]),

    std = np.array([0.229, 0.224, 0.225]))

])

train_transforms = transforms.Compose([transforms.ToPILImage(),

                                        transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),ImageNetPolicy(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])

train_dataset = FoodData(train_data,input_root_dir,256,transforms_train)
batch = 64

valid_size = 0.2

num = train_data.__len__()

# Dividing the indices for train and cross validation

indices = list(range(num))

np.random.shuffle(indices)

split = int(np.floor(valid_size*num))

train_idx,valid_idx = indices[split:], indices[:split]



#Create Samplers

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = DataLoader(train_dataset, batch_size = batch, sampler = train_sampler)

valid_loader = DataLoader(train_dataset, batch_size = batch, sampler = valid_sampler)
# transforms_test = transforms.Compose([

#     transforms.ToPILImage(),

#     transforms.ToTensor(),

#     transforms.Normalize( mean = np.array([0.485, 0.456, 0.406]),

#     std = np.array([0.229, 0.224, 0.225]))

# ])

test_transforms = transforms.Compose([transforms.ToPILImage(),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])



test_data = FoodData(test_data,input_root_dir,256,transform = test_transforms)



test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
dataloaders = {}

dataset_sizes = {}

dataloaders['train'] = train_loader

dataloaders['val'] = valid_loader

dataset_sizes['train'] = train_sampler.__len__()

dataset_sizes['val'] = valid_sampler.__len__()
def mixup_data(x, y, alpha=1.0, use_cuda=True):



    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if alpha > 0.:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1.

    batch_size = x.size()[0]

    if use_cuda:

        index = torch.randperm(batch_size).cuda()

    else:

        index = torch.randperm(batch_size)



    mixed_x = lam * x + (1 - lam) * x[index,:]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam



def mixup_criterion(y_a, y_b, lam):

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

from torch.autograd import Variable

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for data in tqdm(dataloaders[phase]):

                inputs = data['gt'].squeeze(0).to(device)

                labels = data['label'].to(device)

#                 if phase == 'train':

#                   inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.4, device)

#                   inputs, targets_a, targets_b = Variable(inputs), Variable(labels_a), Variable(labels_b)

                  

                inputs = inputs.to(device)

#                 if phase == 'train':

#                   labels_a = labels_a.to(device)

#                   labels_b = labels_b.to(device)

#                 else:

#                   labels = labels.to(device)

                labels = labels.to(device)

                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

#                     loss_func = mixup_criterion(labels_a,labels_b,lam)

                    

                    _, preds = torch.max(outputs, 1)



                    loss = criterion(outputs, labels)

#                     if phase=='train':

#                       loss = loss_func(criterion,outputs)

#                     else:

#                         loss = criterion(outputs,labels) 



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

#                 if phase == 'train':

#                   running_corrects += lam * preds.eq(labels_a.data).sum() + (1 - lam) * preds.eq(labels_b.data).sum()

#                 else:

#                     running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), 'best_model_so_far.pth')



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
import torch

import torch.nn.parallel

import numpy as np

import torch.nn as nn

import torch.nn.functional as F

from IPython import embed



class Downsample(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):

        super(Downsample, self).__init__()

        self.filt_size = filt_size

        self.pad_off = pad_off

        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]

        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]

        self.stride = stride

        self.off = int((self.stride-1)/2.)

        self.channels = channels



        if(self.filt_size==1):

            a = np.array([1.,])

        elif(self.filt_size==2):

            a = np.array([1., 1.])

        elif(self.filt_size==3):

            a = np.array([1., 2., 1.])

        elif(self.filt_size==4):    

            a = np.array([1., 3., 3., 1.])

        elif(self.filt_size==5):    

            a = np.array([1., 4., 6., 4., 1.])

        elif(self.filt_size==6):    

            a = np.array([1., 5., 10., 10., 5., 1.])

        elif(self.filt_size==7):    

            a = np.array([1., 6., 15., 20., 15., 6., 1.])



        filt = torch.Tensor(a[:,None]*a[None,:])

        filt = filt/torch.sum(filt)

        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))



        self.pad = get_pad_layer(pad_type)(self.pad_sizes)



    def forward(self, inp):

        if(self.filt_size==1):

            if(self.pad_off==0):

                return inp[:,:,::self.stride,::self.stride]    

            else:

                return self.pad(inp)[:,:,::self.stride,::self.stride]

        else:

            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])



def get_pad_layer(pad_type):

    if(pad_type in ['refl','reflect']):

        PadLayer = nn.ReflectionPad2d

    elif(pad_type in ['repl','replicate']):

        PadLayer = nn.ReplicationPad2d

    elif(pad_type=='zero'):

        PadLayer = nn.ZeroPad2d

    else:

        print('Pad type [%s] not recognized'%pad_type)

    return PadLayer



class Downsample1D(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):

        super(Downsample1D, self).__init__()

        self.filt_size = filt_size

        self.pad_off = pad_off

        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]

        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]

        self.stride = stride

        self.off = int((self.stride - 1) / 2.)

        self.channels = channels



        # print('Filter size [%i]' % filt_size)

        if(self.filt_size == 1):

            a = np.array([1., ])

        elif(self.filt_size == 2):

            a = np.array([1., 1.])

        elif(self.filt_size == 3):

            a = np.array([1., 2., 1.])

        elif(self.filt_size == 4):

            a = np.array([1., 3., 3., 1.])

        elif(self.filt_size == 5):

            a = np.array([1., 4., 6., 4., 1.])

        elif(self.filt_size == 6):

            a = np.array([1., 5., 10., 10., 5., 1.])

        elif(self.filt_size == 7):

            a = np.array([1., 6., 15., 20., 15., 6., 1.])



        filt = torch.Tensor(a)

        filt = filt / torch.sum(filt)

        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))



        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)



    def forward(self, inp):

        if(self.filt_size == 1):

            if(self.pad_off == 0):

                return inp[:, :, ::self.stride]

            else:

                return self.pad(inp)[:, :, ::self.stride]

        else:

            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])



def get_pad_layer_1d(pad_type):

    if(pad_type in ['refl', 'reflect']):

        PadLayer = nn.ReflectionPad1d

    elif(pad_type in ['repl', 'replicate']):

        PadLayer = nn.ReplicationPad1d

    elif(pad_type == 'zero'):

        PadLayer = nn.ZeroPad1d

    else:

        print('Pad type [%s] not recognized' % pad_type)

    return PadLayer
import torch.nn as nn

import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1, groups=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                 padding=1, groups=groups, bias=False)



def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):

        super(BasicBlock, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if groups != 1:

            raise ValueError('BasicBlock only supports groups=1')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes)

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        if(stride==1):

            self.conv2 = conv3x3(planes,planes)

        else:

            self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),

                conv3x3(planes, planes),)

        self.bn2 = norm_layer(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):

        super(Bottleneck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, planes)

        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes, groups) # stride moved

        self.bn2 = norm_layer(planes)

        if(stride==1):

            self.conv3 = conv1x1(planes, planes * self.expansion)

        else:

            self.conv3 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),

                conv1x1(planes, planes * self.expansion))

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,

                 groups=1, width_per_group=64, norm_layer=None, filter_size=1, pool_only=True):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]

        self.inplanes = planes[0]



        if(pool_only):

            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)

        else:

            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)

        self.bn1 = norm_layer(planes[0])

        self.relu = nn.ReLU(inplace=True)



        if(pool_only):

            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 

                Downsample(filt_size=filter_size, stride=2, channels=planes[0])])

        else:

            self.maxpool = nn.Sequential(*[Downsample(filt_size=filter_size, stride=2, channels=planes[0]), 

                nn.MaxPool2d(kernel_size=2, stride=1), 

                Downsample(filt_size=filter_size, stride=2, channels=planes[0])])



        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)

        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)

        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)

        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):

                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics

                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                else:

                    print('Not initializing')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            # downsample = nn.Sequential(

            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),

            #     norm_layer(planes * block.expansion),

            # )



            downsample = [Downsample(filt_size=filter_size, stride=stride, channels=self.inplanes),] if(stride !=1) else []

            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),

                norm_layer(planes * block.expansion)]

            # print(downsample)

            downsample = nn.Sequential(*downsample)



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, filter_size=filter_size))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, filter_size=filter_size))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)



        return x

def resnet50(pretrained=False, filter_size=1, pool_only=True, **kwargs):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model
!wget https://www.dropbox.com/s/j8bhu6tddbqy5th/resnet50_lpf3-a4e868d2.pth.tar?dl=0 -O resnet50_lpf3.pth.tar
import copy

# model_ft = models.resnet50(pretrained=True)

# model_ft = resnet50(filter_size=3)

# model_ft.load_state_dict(torch.load('resnet50_lpf3.pth.tar')['state_dict'])

# num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

# model_ft.fc = nn.Linear(num_ftrs, 101)

model_ft.load_state_dict(torch.load('best_model_so_far.pth'))

model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.Adam(model_ft.parameters(), lr=0.01, betas=[0.9, 0.999])



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=2)          
!pip install ttach           
import ttach as tta

transforms = tta.Compose(

    [

    # tta.FiveCrops(128,128),   

    tta.HorizontalFlip(),

    tta.VerticalFlip(),

    # tta.FiveCrops(128,128),

    tta.Rotate90(angles=[0, 90]),

#     tta.Scale(scales=[1, 2, 4]),

        

    ]

)

model_ft.load_state_dict(torch.load('best_model_so_far.pth'))

model_ft.eval()

tta_model = tta.ClassificationTTAWrapper(model_ft, transforms,merge_mode='mean')
correct = 0

total = 0

pred_list = []

correct_list = []

with torch.no_grad():

    for images in tqdm(valid_loader):

        data = images['gt'].squeeze(0).to(device)

        target = images['label'].to(device)

        outputs = model_ft(data)

        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)

        pr = predicted.detach().cpu().numpy()

        for i in pr:

          pred_list.append(i)

        tg = target.detach().cpu().numpy()

        for i in tg:

          correct_list.append(i)

        correct += (predicted == target).sum().item()



print('Accuracy of the network on the 10000 test images: %f %%' % (

    100 * correct / total))