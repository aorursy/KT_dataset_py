# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import csv

from PIL import Image

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms



import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

from torch.utils.data.sampler import SubsetRandomSampler



import torch.nn as nn

import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torch.utils.data import sampler

import torchvision

import torchvision.datasets as dset

import torchvision.transforms as T

import timeit

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.autograd import Function

from torch.nn.modules.module import Module

from torch.nn.parameter import Parameter

from torch.nn.functional import conv2d

import torch.nn.functional as F

import numpy as np





class Kerv2d(nn.Conv2d):

    '''

    kervolution with following options:

    kernel_type: [linear, polynomial, gaussian, etc.]

    default is convolution:

             kernel_type --> linear,

    balance, power, gamma is valid only when the kernel_type is specified

    if learnable_kernel = True,  they just be the initial value of learable parameters

    if learnable_kernel = False, they are the value of kernel_type's parameter

    the parameter [power] cannot be learned due to integer limitation

    '''

    def __init__(self, in_channels, out_channels, kernel_size, 

            stride=1, padding=0, dilation=1, groups=1, bias=True,

            kernel_type='linear', learnable_kernel=False, kernel_regularizer=False,

            balance=1, power=3, gamma=1):



        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.kernel_type = kernel_type

        self.learnable_kernel, self.kernel_regularizer = learnable_kernel, kernel_regularizer

        self.balance, self.power, self.gamma = balance, power, gamma



        # parameter for kernel type

        if learnable_kernel == True:

            self.balance = nn.Parameter(torch.cuda.FloatTensor([balance] * out_channels), requires_grad=True).view(-1, 1)

            self.gamma   = nn.Parameter(torch.cuda.FloatTensor([gamma]   * out_channels), requires_grad=True).view(-1, 1)



    def forward(self, input):



        minibatch, in_channels, input_width, input_hight = input.size()

        assert(in_channels == self.in_channels)

        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        input_unfold = input_unfold.view(minibatch, 1, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, -1)

        weight_flat  = self.weight.view(self.out_channels, -1, 1)

        output_width = (input_width - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1

        output_hight = (input_hight - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1



        if self.kernel_type == 'linear':

            output = (input_unfold * weight_flat).sum(dim=2)



        elif self.kernel_type == 'manhattan':

            output = -((input_unfold - weight_flat).abs().sum(dim=2))



        elif self.kernel_type == 'euclidean':

            output = -(((input_unfold - weight_flat)**2).sum(dim=2))



        elif self.kernel_type == 'polynomial':

            output = ((input_unfold * weight_flat).sum(dim=2) + self.balance)**self.power



        elif self.kernel_type == 'gaussian':

            output = (-self.gamma*((input_unfold - weight_flat)**2).sum(dim=2)).exp() + 0



        else:

            raise NotImplementedError(self.kernel_type+' kervolution not implemented')



        if self.bias is not None:

            output += self.bias.view(self.out_channels, -1)



        return output.view(minibatch, self.out_channels, output_width, output_hight)





class Kerv1d(nn.Conv1d):

    r"""Applies a 1D kervolution over an input signal composed of several input

        planes.

        Args:

            in_channels (int): Number of channels in the input image

            out_channels (int): Number of channels produced by the convolution

            kernel_size (int or tuple): Size of the convolving kernel

            stride (int or tuple, optional): Stride of the convolution. Default: 1

            padding (int or tuple, optional): Zero-padding added to both sides of

                the input. Default: 0

            padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`

            dilation (int or tuple, optional): Spacing between kernel

                elements. Default: 1

            groups (int, optional): Number of blocked connections from input

                channels to output channels. Default: 1

            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

            kernel_type (str), Default: 'linear'

            learnable_kernel (bool): Learnable kernel parameters.  Default: False 

            balance=1, power=3, gamma=1

        Shape:

            - Input: :math:`(N, C_{in}, L_{in})`

            - Output: :math:`(N, C_{out}, L_{out})` where

            .. math::

                L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}

                            \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

        Examples::

            >>> m = Kerv1d(16, 33, 3, kernel_type='polynomial', learnable_kernel=True)

            >>> input = torch.randn(20, 16, 50)

            >>> output = m(input)

        .. _kervolution:

            https://arxiv.org/pdf/1904.03955.pdf

        """



    def __init__(self, in_channels, out_channels, kernel_size, 

            stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',

            kernel_type='linear', learnable_kernel=False, balance=1, power=3, gamma=1):



        super(Kerv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.kernel_type, self.learnable_kernel = kernel_type, learnable_kernel

        self.balance, self.power, self.gamma = balance, power, gamma

        self.unfold = nn.Unfold((kernel_size,1), (dilation,1), (padding, 0), (stride,1))



        # parameter for kernels

        # if learnable_kernel == True:

        # self.balance = nn.Parameter(torch.FloatTensor([balance] * out_channels)).view(-1, 1)

        # self.gamma   = nn.Parameter(torch.FloatTensor([gamma]   * out_channels)).view(-1, 1)





    def forward(self, input):

        input = self.unfold(input.unsqueeze(-1)).unsqueeze(1)

        weight  = self.weight.view(self.out_channels, -1, 1)



        if self.kernel_type == 'linear':

            output = (input * weight).sum(dim=2)



        elif self.kernel_type == 'manhattan':

            output = -((input - weight).abs().sum(dim=2))



        elif self.kernel_type == 'euclidean':

            output = -(((input - weight)**2).sum(dim=2))



        elif self.kernel_type == 'polynomial':

            output = ((input * weight).sum(dim=2) + self.balance)**self.power



        elif self.kernel_type == 'gaussian':

            output = (-self.gamma*((input - weight)**2).sum(dim=2)).exp() + 0



        else:

            raise NotImplementedError(self.kernel_type+' Kerv1d not implemented')



        if self.bias is not None:

            output += self.bias.view(self.out_channels, -1)



        return output





    def cuda(self, device=None):

        if self.learnable_kernel == True:

            self.balance = self.balance.cuda(device)

            self.gamma = self.gamma.cuda(device)

        return self._apply(lambda t: t.cuda(device))



class LinearKernel(torch.nn.Module):

    def __init__(self):

        super(LinearKernel, self).__init__()

    

    def forward(self, x_unf, w, b):

        t = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)

        if b is not None:

            return t + b

        return t

        

        

class PolynomialKernel(LinearKernel):

    def __init__(self, cp=2.0, dp=3, train_cp=True):

        super(PolynomialKernel, self).__init__()

        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=train_cp))

        self.dp = dp



    def forward(self, x_unf, w, b):

        return (self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp





class GaussianKernel(torch.nn.Module):

    def __init__(self, gamma):

        super(GaussianKernel, self).__init__()

        self.gamma = torch.nn.parameter.Parameter(

                            torch.tensor(gamma, requires_grad=True))

    

    def forward(self, x_unf, w, b):

        l = x_unf.transpose(1, 2)[:, :, :, None] - w.view(1, 1, -1, w.size(0))

        l = torch.sum(l**2, 2)

        t = torch.exp(-self.gamma * l)

        if b:

            return t + b

        return t

        

       

class KernelConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=PolynomialKernel,

                 stride=1, padding=0, dilation=1, groups=1, bias=None,

                 padding_mode='zeros'):

        '''

        Follows the same API as torch Conv2d except kernel_fn.

        kernel_fn should be an instance of the above kernels.

        '''

        super(KernelConv2d, self).__init__(in_channels, out_channels, 

                                           kernel_size, stride, padding,

                                           dilation, groups, bias, padding_mode)

        self.kernel_fn = kernel_fn()

   

    def compute_shape(self, x):

        h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

        w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        return h, w

    

    def forward(self, x):

        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation,self.padding, self.stride)

        h, w = self.compute_shape(x)

        return self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)
print(os.listdir("../input/kernvolution-pytorch-library/kervolution-pytorch-master"))
import sys

sys.path.append("../input/kernvolution-pytorch-library/kervolution-pytorch-master/")

from layer import KernelConv2d, GaussianKernel, PolynomialKernel

class CancerDataset(Dataset):

    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.ToTensor()]), labels_dict={}):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.transform = transform

        self.labels_dict = labels_dict

        if self.datatype == 'train':

            lab = [labels_dict[i.split('.')[0]] for i in self.image_files_list]

            print(lab)

            self.labels = lab 

        else:

            self.labels = [0 for _ in range(len(self.image_files_list))]



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = os.path.join(self.datafolder, self.image_files_list[idx])

        image = Image.open(img_name)

        image = self.transform(image)

        img_name_short = self.image_files_list[idx].split('.')[0]



        if self.datatype == 'train':

            label = self.labels_dict[img_name_short]

        else:

            label = 0

        return image, label


import pandas as pd

import os

os.listdir("../input/histopathologic-cancer-detection")

#pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
IMAGE_NOT_FOUND_COUNTER = 0



labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')



data_transforms = transforms.Compose([

    #transforms.CenterCrop(32),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

data_transforms_test = transforms.Compose([

    #transforms.CenterCrop(32),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])







tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.2)

print("number of training data: ",len(tr))

print("number of testing  data: ",len(val))

# dictionary with labels and ids of train data

img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}



train_sampler = SubsetRandomSampler(list(tr.index))

valid_sampler = SubsetRandomSampler(list(val.index))

batch_size = 128

num_workers = 0



dataset = CancerDataset(datafolder='../input/histopathologic-cancer-detection/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)

test_set = CancerDataset(datafolder='../input/histopathologic-cancer-detection/test/', datatype='test', transform=data_transforms_test)

# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
class Flatten(nn.Module):

    def forward(self, x):

        N, C, H, W = x.size() # read in N, C, H, W

        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
avg_loss_list = []

acc_list = []



def train(model, train_loader ,loss_fn, optimizer, num_epochs = 1):

    total_loss =0



    for epoch in range(num_epochs):

        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))

        model.train()

        for t, (x, y) in enumerate(train_loader):

            x_var = Variable(x.type(gpu_dtype))

            y_var = Variable(y.type(gpu_dtype).long())



            scores = model(x_var)

            loss = loss_fn(scores, y_var)

            total_loss += loss.data

            

            if (t + 1) % print_every == 0:

                avg_loss = total_loss/print_every

                print('t = %d, avg_loss = %.4f' % (t + 1, avg_loss) )

                avg_loss_list.append(avg_loss)

                total_loss = 0

                



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        acc = check_accuracy(fixed_model_gpu, valid_loader)

        print('acc = %f' %(acc))

            

def check_accuracy(model, loader):

    print('Checking accuracy on test set')   

    num_correct = 0

    num_samples = 0

    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)

    for x, y in loader:

        x_var = Variable(x.type(gpu_dtype))



        scores = model(x_var)

        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()

        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples

    acc_list.append(acc)

    return acc

    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
# class KernelConv2d(torch.nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=PolynomialKernel,

#                  stride=1, padding=0, dilation=1, groups=1, bias=None,

#                  padding_mode='zeros'):

#         '''

#         Follows the same API as torch Conv2d except kernel_fn.

#         kernel_fn should be an instance of the above kernels.

#         '''

        

#         super(KernelConv2d, self).__init__(in_channels, out_channels, 

#                                            kernel_size=kernel_size, stride=stride, padding=1, bias=False)

#         self.kernel_fn = kernel_fn()

   

#     def compute_shape(self, x):

#         h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

#         w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

#         return h, w

    

#     def forward(self, x):

#         x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation,self.padding, self.stride)

#         h, w = self.compute_shape(x)

#         return self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)
from torchvision import models



print_every = 20

gpu_dtype = torch.cuda.FloatTensor



out_1 = 32

out_2 = 64

out_3 = 128

out_4 = 256



k_size_1 = 3

padding_1 = 1





num_epochs = 6







fixed_model_base = nn.Sequential( # You fill this in!

    #KernelConv2d(3, out_1, kernel_size=k_size_1),

                Kerv2d(3, out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # out_1-k_size_1+1 = 26

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

    #KernelConv2d(out_1, out_1, kernel_size=k_size_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), #26 - 4 + 1 = 23

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

    #KernelConv2d(out_1, out_1, kernel_size=k_size_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # 23 -3 = 20

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

    

                nn.MaxPool2d(2, stride=2),

    #KernelConv2d(out_1, out_2, kernel_size=k_size_1),

                nn.Conv2d(out_1 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 20 -3 = 17

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_2),

    #KernelConv2d(out_2, out_2, kernel_size=k_size_1),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True), 

                nn.BatchNorm2d(out_2),

    #KernelConv2d(out_2, out_2, kernel_size=k_size_1),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_2),

    

                nn.MaxPool2d(2, stride=2),

    #KernelConv2d(out_2, out_3, kernel_size=k_size_1),

                nn.Conv2d(out_2 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1),

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

    #KernelConv2d(out_3, out_3, kernel_size=k_size_1),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

    #KernelConv2d(out_3, out_3, kernel_size=k_size_1),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

    

                nn.MaxPool2d(2, stride=2),

    #KernelConv2d(out_3, out_4, kernel_size=k_size_1),

                nn.Conv2d(out_3 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

    #KernelConv2d(out_4, out_4, kernel_size=k_size_1),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

    #KernelConv2d(out_4, out_4, kernel_size=k_size_1),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

                        

    

                nn.MaxPool2d(2, stride=2), #17/2 = 7

                Flatten(),

                

                nn.Linear(9216,512 ), # affine layer

                nn.ReLU(inplace=True),

                nn.Linear(512,10), # affine layer

                nn.ReLU(inplace=True),

                nn.Linear(10,2), # affine layer

            )

fixed_model_gpu = fixed_model_base.type(gpu_dtype)

print(fixed_model_gpu)

loss_fn = nn.modules.loss.CrossEntropyLoss()

optimizer = optim.RMSprop(fixed_model_gpu.parameters(), lr = 1e-3)



train(fixed_model_gpu, train_loader ,loss_fn, optimizer, num_epochs=num_epochs)

check_accuracy(fixed_model_gpu, valid_loader)
print(avg_loss_list,acc_list)
import matplotlib.pyplot as plt



plt.plot([i+1 for i in range((len(acc_list)))],acc_list)

plt.show()
print("Loss")

plt.plot([print_every*batch_size*(i+1)/len(tr) for i in range((len(avg_loss_list)))],avg_loss_list)
fixed_model_gpu.eval()

preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = fixed_model_gpu(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

        

test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})



test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])



data_to_submit = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

data_to_submit = pd.merge(data_to_submit, test_preds, left_on='id', right_on='imgs')

data_to_submit = data_to_submit[['id', 'preds']]

data_to_submit.columns = ['id', 'label']

data_to_submit.head()
data_to_submit.to_csv('csv_to_submit_knn.csv', index = False)
print("KNN without ReLu and Pooling")
print_every = 20

gpu_dtype = torch.cuda.FloatTensor



out_1 = 32

out_2 = 64

out_3 = 128

out_4 = 256



k_size_1 = 3

padding_1 = 1





num_epochs = 6







fixed_model_base = nn.Sequential( 

                Kerv2d(3, out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # out_1-k_size_1+1 = 26

               

                nn.BatchNorm2d(out_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), #26 - 4 + 1 = 23

                nn.BatchNorm2d(out_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # 23 -3 = 20

                nn.BatchNorm2d(out_1),

    

                nn.AvgPool2d(2, stride=2),

                nn.Conv2d(out_1 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 20 -3 = 17

                nn.BatchNorm2d(out_2),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_2),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_2),

    

                nn.AvgPool2d(2, stride=2),

                nn.Conv2d(out_2 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1),

                nn.BatchNorm2d(out_3),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_3),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_3),

                nn.AvgPool2d(2, stride=2),

                nn.Conv2d(out_3 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_4),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_4),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.BatchNorm2d(out_4),

                        

    

                nn.AvgPool2d(2, stride=2), #17/2 = 7

                Flatten(),

                

#                 nn.Linear(9216,512 ), # affine layer

#                 nn.ReLU(inplace=True),

#                 nn.Linear(512,10), # affine layer

#                 nn.ReLU(inplace=True),

#                 nn.Linear(10,2), # affine layer

            )

fixed_model_gpu = fixed_model_base.type(gpu_dtype)

print(fixed_model_gpu)

loss_fn = nn.modules.loss.CrossEntropyLoss()

optimizer = optim.RMSprop(fixed_model_gpu.parameters(), lr = 1e-3)



train(fixed_model_gpu, train_loader ,loss_fn, optimizer, num_epochs=num_epochs)

check_accuracy(fixed_model_gpu, valid_loader)
print(avg_loss_list,acc_list)
plt.plot([i+1 for i in range((len(acc_list)))],acc_list)

plt.show()
print("Loss")

plt.plot([print_every*batch_size*(i+1)/len(tr) for i in range((len(avg_loss_list)))],avg_loss_list)
fixed_model_gpu.eval()

preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = fixed_model_gpu(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

        

test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})



test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])



data_to_submit = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

data_to_submit = pd.merge(data_to_submit, test_preds, left_on='id', right_on='imgs')

data_to_submit = data_to_submit[['id', 'preds']]

data_to_submit.columns = ['id', 'label']

data_to_submit.head()
data_to_submit.to_csv('csv_to_submit_knn_without_relu_pool.csv', index = False)
print("Custom CNN:")
print_every = 20

gpu_dtype = torch.cuda.FloatTensor



out_1 = 32

out_2 = 64

out_3 = 128

out_4 = 256



k_size_1 = 3

padding_1 = 1





num_epochs = 6







fixed_model_base = nn.Sequential( 

                nn.Conv2d(3, out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # out_1-k_size_1+1 = 26

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), #26 - 4 + 1 = 23

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

                nn.Conv2d(out_1 , out_1, padding= padding_1, kernel_size=k_size_1, stride=1), # 23 -3 = 20

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_1),

    

                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(out_1 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 20 -3 = 17

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_2),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True), 

                nn.BatchNorm2d(out_2),

                nn.Conv2d(out_2 , out_2, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_2),

    

                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(out_2 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1),

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

                nn.Conv2d(out_3 , out_3, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_3),

                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(out_3 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

                nn.Conv2d(out_4 , out_4, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(out_4),

                        

    

                nn.MaxPool2d(2, stride=2), #17/2 = 7

                Flatten(),

                

                nn.Linear(9216,512 ), # affine layer

                nn.ReLU(inplace=True),

                nn.Linear(512,10), # affine layer

                nn.ReLU(inplace=True),

                nn.Linear(10,2), # affine layer

            )

fixed_model_gpu = fixed_model_base.type(gpu_dtype)

print(fixed_model_gpu)

loss_fn = nn.modules.loss.CrossEntropyLoss()

optimizer = optim.RMSprop(fixed_model_gpu.parameters(), lr = 1e-3)



train(fixed_model_gpu, train_loader ,loss_fn, optimizer, num_epochs=num_epochs)

check_accuracy(fixed_model_gpu, valid_loader)
print(avg_loss_list,acc_list)
plt.plot([i+1 for i in range((len(acc_list)))],acc_list)

plt.show()
print("Loss")

plt.plot([print_every*batch_size*(i+1)/len(tr) for i in range((len(avg_loss_list)))],avg_loss_list)
fixed_model_gpu.eval()

preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = fixed_model_gpu(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

        

test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})



test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])



data_to_submit = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

data_to_submit = pd.merge(data_to_submit, test_preds, left_on='id', right_on='imgs')

data_to_submit = data_to_submit[['id', 'preds']]

data_to_submit.columns = ['id', 'label']

data_to_submit.head()
data_to_submit.to_csv('csv_to_submit_cnn.csv', index = False)