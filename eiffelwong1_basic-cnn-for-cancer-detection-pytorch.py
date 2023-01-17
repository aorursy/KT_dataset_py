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

class CancerDataset(Dataset):

    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.ToTensor()]), labels_dict={}):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.transform = transform

        self.labels_dict = labels_dict

        if self.datatype == 'train':

            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]

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
IMAGE_NOT_FOUND_COUNTER = 0



labels = pd.read_csv('../input/train_labels.csv')



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







tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.1)

print("number of training data: ",len(tr))

print("number of testing  data: ",len(val))

# dictionary with labels and ids of train data

img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}



train_sampler = SubsetRandomSampler(list(tr.index))

valid_sampler = SubsetRandomSampler(list(val.index))

batch_size = 256

num_workers = 0



dataset = CancerDataset(datafolder='../input/train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)

test_set = CancerDataset(datafolder='../input/test/', datatype='test', transform=data_transforms_test)

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

    

                #nn.Conv2d(out_11 , out_12, padding= padding_1, kernel_size=k_size_1, stride=1), # 17 -3 = 14

                #nn.ReLU(inplace=True),

                #nn.BatchNorm2d(out_12),

    

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



plt.plot([print_every*batch_size*(i+1)/len(tr) for i in range((len(avg_loss_list)))],avg_loss_list)

plt.plot([i+1 for i in range((len(acc_list)))],acc_list)

plt.show()
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



data_to_submit = pd.read_csv('../input/sample_submission.csv')

data_to_submit = pd.merge(data_to_submit, test_preds, left_on='id', right_on='imgs')

data_to_submit = data_to_submit[['id', 'preds']]

data_to_submit.columns = ['id', 'label']

data_to_submit.head()
data_to_submit.to_csv('csv_to_submit.csv', index = False)