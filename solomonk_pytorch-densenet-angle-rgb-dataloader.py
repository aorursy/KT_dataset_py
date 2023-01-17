# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%reset -f



import torch

from torch.autograd import Variable

import numpy as np

import pandas

import numpy as np

import pandas as pd

from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

import logging

import numpy

import numpy as np

from __future__ import print_function

from __future__ import division

import math

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import os

import torch

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from torch import nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

import time

from sklearn.preprocessing import PolynomialFeatures

import pandas as pd

import numpy as np

import scipy

%matplotlib inline

from pylab import rcParams



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from torch import nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook

import seaborn as sns

# %%timeit

use_cuda = torch.cuda.is_available()

# use_cuda = False



print("USE CUDA=" + str (use_cuda))

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor

# fix seed

seed=17*19

np.random.seed(seed)

torch.manual_seed(seed)

if use_cuda:

    torch.cuda.manual_seed(seed)
import os

import sys

import random

import numpy as np

from torchvision import transforms

from PIL import Image

import torch

import torch.utils.data as data



IMG_EXTENSIONS = [

    '.jpg',

    'png'

]



to_tensor = transforms.Compose([transforms.ToTensor()])



def is_img_file(filename):

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



# def default_loader(path):

# 	return Image.open(path).convert('RGB')



def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):

    img = Image.open(filename).convert('RGB')

    if size is not None:

        if keep_asp:

            size2 = int(size * 1.0 / img.size[0] * img.size[1])

            img = img.resize((size, size2), Image.ANTIALIAS)

        else:

            img = img.resize((size, size), Image.ANTIALIAS)



    elif scale is not None:

        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    img = np.array(img).transpose(2, 0, 1)

    img = torch.from_numpy(img).float()

    return img





def default_loader_scale(input_path, size=20):

    input_image = (Image.open(input_path)).convert('RGB')

    if size is not None:

        input_image = input_image.resize((size, size), Image.ANTIALIAS)



    # input_image = np.array(input_image).transpose(2, 0, 1)

    # input_image = torch.from_numpy(input_image).float()

    return input_image



def default_loader(input_path):

    # pil_to_tensor = transforms.ToTensor()

    input_image = (Image.open(input_path)).convert('RGB')

    # input_image = pil_to_tensor(input_image)

    # img = np.array(img).transpose(2, 0, 1)

    # img = torch.from_numpy(img).float()

    return input_image



def find_classes(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx



def make_dataset(dir, class_to_idx):

    tensors = []

    if not os.path.exists(dir):

        print("Seed dataset %s doesn't exist." % (dir))

        sys.exit()

    else:

        dir = os.path.expanduser(dir)

        for target in sorted(os.listdir(dir)):

            d = os.path.join(dir, target)

            if not os.path.isdir(d):

                continue



            for root, _, fnames in sorted(os.walk(d)):

                for fname in sorted(fnames):

                    if is_img_file(fname):

                        path = os.path.join(root, fname)

                        item = (path, class_to_idx[target])

                        tensors.append(item)



    return tensors





class SeedImageDataset(data.Dataset):

    def __init__(self,root,phase,loader=default_loader_scale,transform=None):



        classes, class_to_idx = find_classes(root)

        print ('Classes: {}'.format(classes))

        print ('# Classes: {}'.format(len(classes)))

        print ('Class to idx: {}'.format(class_to_idx))



        tensors = make_dataset(root, class_to_idx)

        if len(tensors) == 0:

            raise (RuntimeError(

                "Found 0 sound files in subfolders of: " + root + "Supported img file extensions are: " + ",".join(

                    IMG_EXTENSIONS)))



        self.tensors = tensors

        self.root = root

        self.phase = phase

        self.classes = classes

        self.class_to_idx = class_to_idx

        self.transform = transform

        self.loader = loader





    def __getitem__(self, index):

        # Get path of input image and ground truth

        input, target = self.tensors[index]

        # Acquire input image and ground truth

        input_tensor = self.loader(input)

        # print (type(input_tensor)) # <class 'PIL.Image.Image'>



        if self.transform is not None:

            input_tensor = self.transform(input_tensor)



        if type(input_tensor) is not torch.FloatTensor:

        # print (type(input_tensor)) # MUST BE <class 'torch.FloatTensor'>

            input_tensor=to_tensor(input_tensor)



        return input_tensor, target



    def __len__(self):

        return len(self.tensors)

#your custom aug function for numpy image:

#seems like all flip augmentations may decrease performance





normalize = transforms.Normalize(

   mean=[0.485, 0.456, 0.406],

   std=[0.229, 0.224, 0.225]

)

preprocess = transforms.Compose([

#    transforms.Scale(256),

#    transforms.CenterCrop(224),

   transforms.ToTensor(),

   normalize

])



train = pd.read_json('../input/train.json')

train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

train['band_1'] = train['band_1'].apply(lambda x: np.array(x).reshape(75, 75))

train['band_2'] = train['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

        

batch_size = 32

# train_ds = ImageDataset(train, include_target = True, u =0.5, X_transform = random_vertical_flip)

train_ds = ImageDataset(train, include_target = True, X_transform = preprocess)

USE_CUDA = False #for kernel

THREADS = 1 #for kernel

train_loader = data.DataLoader(train_ds, batch_size,

                                    sampler = RandomSampler(train_ds),

                                    num_workers = THREADS,

                                    pin_memory= USE_CUDA )

                                    

#prseudo code for train

# for i, dict_ in enumerate(train_loader):

#     images  = dict_['img']

#     target  = dict_['target'].type(torch.FloatTensor)

    

#     if USE_CUDA:

#         images = images.cuda()

#         target = target.cuda()

    

#     images = Variable(images)

#     target = Variable(target)    

    

#     #for kernel:

#     print(target)

#     if i ==0 : break

%matplotlib inline

from torchvision.utils import make_grid

    

def flaotTensorToImage(img, mean=0, std=1):

    """convert a tensor to an image"""

    img = img.numpy()

    img= img.reshape(75, 75, 3)    

    img = (img*std+ mean)*255

    img = img.astype(np.uint8)    

#     print (img.shape)

    return img    

    

import matplotlib.pyplot as plt



imagesToShow=4



for i, data in enumerate(train_loader, 0):

    print('i=%d: '%(i))            

    images  = data['img']

    target  = data['target'].type(torch.FloatTensor)

    num = len(images)

    ax = plt.subplot(1, imagesToShow, i + 1)

    plt.tight_layout()

    ax.set_title('Sample #{}'.format(i))

    ax.axis('off')

    

    for n in range(num):

        image=images[n]

        label=target[n]

        plt.imshow (flaotTensorToImage(image))

#         show(image)

        

    if i==imagesToShow-1:

        break    

from torch.utils.data import TensorDataset, DataLoader



class FullTrainningDataset(torch.utils.data.Dataset):

    def __init__(self, full_ds, offset, length):

        self.full_ds = full_ds

        self.offset = offset

        self.length = length

        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")

        super(FullTrainningDataset, self).__init__()

        

    def __len__(self):        

        return self.length

    

    def __getitem__(self, i):

        return self.full_ds[i+self.offset]

    

validationRatio=0.11    



def trainTestSplit(dataset, val_share=0.11):

    val_offset = int(len(dataset)*(1-val_share))

    print ("Offest:" + str(val_offset))

    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, 

                                                                              val_offset, len(dataset)-val_offset)



train_ds, val_ds = trainTestSplit(train_loader)

t_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,

                                            num_workers=1)

v_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)



print (t_loader)

print (v_loader)
import sys

import math



class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):

        super(Bottleneck, self).__init__()

        interChannels = 4*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)

        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,

                               bias=False)

        self.bn2 = nn.BatchNorm2d(interChannels)

        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,

                               padding=1, bias=False)



    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))

        out = self.conv2(F.relu(self.bn2(out)))

        out = torch.cat((x, out), 1)

        return out



class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):

        super(SingleLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(nChannels)

        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,

                               padding=1, bias=False)



    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))

        out = torch.cat((x, out), 1)

        return out



class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):

        super(Transition, self).__init__()

        self.bn1 = nn.BatchNorm2d(nChannels)

        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,

                               bias=False)



    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))

        out = F.avg_pool2d(out, 2)

        return out





class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):

        super(DenseNet, self).__init__()



        nDenseBlocks = (depth-4) // 3

        if bottleneck:

            nDenseBlocks //= 2



        nChannels = 2*growthRate

        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,

                               bias=False)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)

        nChannels += nDenseBlocks*growthRate

        nOutChannels = int(math.floor(nChannels*reduction))

        self.trans1 = Transition(nChannels, nOutChannels)



        nChannels = nOutChannels

        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)

        nChannels += nDenseBlocks*growthRate

        nOutChannels = int(math.floor(nChannels*reduction))

        self.trans2 = Transition(nChannels, nOutChannels)



        nChannels = nOutChannels

        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)

        nChannels += nDenseBlocks*growthRate



        self.bn1 = nn.BatchNorm2d(nChannels)

        self.fc = nn.Linear(128, nClasses)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                m.bias.data.zero_()



    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):

        layers = []

        for i in range(int(nDenseBlocks)):

            if bottleneck:

                layers.append(Bottleneck(nChannels, growthRate))

            else:

                layers.append(SingleLayer(nChannels, growthRate))

            nChannels += growthRate

        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.conv1(x)

        out = self.trans1(self.dense1(out))

        out = self.trans2(self.dense2(out))

        out = self.dense3(out)

        # print(out.data.shape)

        out = F.avg_pool2d(F.relu(self.bn1(out)), 8)

        out = out.view(out.size(0), -1)

        # print(out.data.shape)

        out = F.sigmoid(self.fc(out))

        return out



model = DenseNet(growthRate=8, depth=20, reduction=0.5,

                            bottleneck=True, nClasses=1)



print (model)



print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss



# NN params

LR = 0.0005

MOMENTUM= 0.95

optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization

if use_cuda:    

    model.cuda()

    loss_func.cuda()



print(optimizer)

print(loss_func)





import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.autograd import Variable



criterion = loss_func

all_losses = []

val_losses = []

num_epoches=10



if __name__ == '__main__':



    for epoch in range(num_epoches):

        print('Epoch {}'.format(epoch + 1))

        print('*' * 5 + ':')

        running_loss = 0.0

        running_acc = 0.0

        

        for i, data in enumerate(train_loader, 0):        

            img  = data['img']

            label  = data['target'].type(torch.FloatTensor)



#         for i, data in enumerate(train_loader, 0):    

#             img  = data['img']

#             label  = data['target']#     

            

            img, label = Variable(img), Variable(label)  # RuntimeError: expected CPU tensor (got CUDA tensor)

    

            out = model(img).type(torch.FloatTensor).squeeze(1)

            loss = criterion(out, label)

            running_loss += loss.data[0] * label.size(0)

    

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()               

    

        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_ds))))

    

#         model.eval()

#         eval_loss = 0

#         eval_acc = 0

#         for data in v_loader:            

#             img  = data['img']

#             label  = data['target']

            

#             img = Variable(img, volatile=True)

#             label = Variable(label, volatile=True)

    

#             out = model(img).type(torch.FloatTensor).squeeze(1)

#             loss = criterion(out, label)

#             eval_loss += loss.data[0] * label.size(0)

    

#         print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))

#         val_losses.append(eval_loss / (len(val_ds)))

#         print()

    

    torch.save(model.state_dict(), './cnn.pth')