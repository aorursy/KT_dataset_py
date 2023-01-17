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
!ls ..
import PIL
from skimage import io, transform
import torch

import torchvision
import cv2

from collections import OrderedDict
import matplotlib.pyplot as plt

from PIL import Image

from skimage import io
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils, models

from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Assuming that we are on a CUDA machine, this should print a CUDA device:



print(device)
root_dir = '../input/imet-2019-fgvc6/'
labels = pd.read_csv(root_dir+'labels.csv')

df_train = pd.read_csv(root_dir+'train.csv')

df_sub = pd.read_csv(root_dir+'sample_submission.csv')
print(df_train.shape)

df_train.head()
labels.head()
labels.shape
n_categories = len(labels['attribute_name'].unique())

print(n_categories)
img_name = df_train.iloc[65, 0]

plt.imshow(cv2.cvtColor(cv2.imread(root_dir+'train/'+img_name+'.png'), cv2.COLOR_BGR2RGB))

print(df_train.iloc[65, 1])
df_train.iloc[3, 1].split(' ')
arr = list(map(int, df_train.iloc[3, 1].split(' ')))

print(type(arr[0]))
arr =[]

for i in range(df_train.shape[0]):

    arr.append(list(map(int, df_train.iloc[i, 1].split(' '))))

df_train['attributes_int'] = arr
class IMetDataset(Dataset):

    def __init__(self, label_file, 

                 train_csv, 

                 train_dir,

                 transform=None):

        

        self.label_file = label_file

        self.train_df = train_csv

        self.train_dir = train_dir

        self.transform = transform

        

        

    def __len__(self):

        return self.train_df.shape[0]

        

    def __getitem__(self, idx):

        img_name = os.path.join(self.train_dir, self.train_df.iloc[idx, 0]+'.png')

        

        #img = io.imread(img_name)

         

        img = cv2.imread(img_name)

        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB) 

        img=img.transpose((2, 0, 1)) 

        

        if self.transform:

            img = self.transform(img)

        

        labs = self.train_df.iloc[idx, 2]

      #  print("Labels: ", labs)

        ans = np.zeros((1103, 1))

        for label in labs:

            ans[label]=1

        #print(ans.shape)

     #   print("One hot indices: ", np.where(ans==1)[0])

        

        return [img, ans]


class IMetTestDataset(Dataset):

    def __init__(self, test_dir, transformations=None):

        self.test_dir =  test_dir

        self.img_list = os.listdir(root_dir+'test')

        self.transform = transformations

            

    def __len__(self):

        'Denotes the total number of samples'

        return len(self.img_list)



    def __getitem__(self, idx):

        'Generates one sample of data'

        # Select sample

        img_loc = os.path.join(self.test_dir, self.img_list[idx])

      #  print(img_loc)

      #  img = io.imread(img_loc)

        

        img = cv2.imread(img_loc)

        img = cv2.cvtColor(cv2.imread(img_loc), cv2.COLOR_BGR2RGB) 

        img=img.transpose((2, 0, 1)) 

        

        if self.transform:

            img = self.transform(img)

        

        img_name = self.img_list[idx].split('.')[0]

        

        return [img, img_name]
transformations = transforms.Compose([

                                transforms.ToPILImage(),

                                transforms.Resize((224, 224)),

                                transforms.RandomHorizontalFlip(),

                                transforms.ToTensor(),

                                transforms.Normalize(

                                    [0.485, 0.456, 0.406], 

                                    [0.229, 0.224, 0.225]

                                )

                            ])



train_dataset = IMetDataset(labels, df_train, root_dir+'train/', 

                            transformations)



test_dataset = IMetTestDataset(root_dir+'test/', transformations)
trainloader = DataLoader(train_dataset, batch_size=64,

                        shuffle=True, num_workers=0)

testloader = DataLoader(test_dataset, batch_size=1,

                        shuffle=False, num_workers=0)
plt.imshow(train_dataset[1001][0].numpy().transpose(1, 2, 0))
os.listdir("../input/vgg16-pytorch/")
model = models.vgg16(pretrained=False)

model.load_state_dict(torch.load("../input/vgg16-pytorch/vgg16-397923af.pth"))

model
model.classifier[-1] = nn.Linear(in_features=4096, out_features=n_categories)

model.classifier.add_module('sigmoid', nn.Sigmoid())

model.classifier.named_parameters
for param in model.parameters():

    param.require_grad=False

for param in model.classifier.parameters():

    param.require_grad=True

for param in model.features[-1: -4]:

    param.require_grad=True
#model.load_state_dict(torch.load("../models/model1.pth"))
model.to(device)
import torch.optim as optim

from torch.autograd import Variable

criterion = nn.BCELoss(reduction='mean').to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)
def show_cuda():

    if device.type == 'cuda':

        print(torch.cuda.get_device_name(0))

        print('Memory Usage:')

        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

show_cuda()
for epoch in range(15):

    print("Epoch", epoch, "Started...")

    running_loss=0

    for i, data in enumerate(trainloader, 0):

        optimizer.zero_grad()

        

        images, label = data

        

        inputs = images.type(torch.FloatTensor)

        '''

        inputs=Variable(inputs).cuda()

     #   labels = labels.view()

        label = Variable(label).cuda()

        '''

        inputs, label = Variable(inputs.to(device)), Variable(label.to(device))

        

        outputs = model(inputs)

        '''  

        print(outputs.shape)

        print(label.shape)

        

        print(type(outputs))

        print(type(label))

        

        '''

        loss=criterion(outputs.type(torch.FloatTensor), label.type(torch.FloatTensor))

        

        loss.backward()

        optimizer.step()

        loss=loss.item()

        running_loss+=float(loss)

    #    if running_loss<1.5:

    #        break

        if i%200==0:

            print("Epoch: ", epoch,"  Running Loss:",running_loss)

            

            running_loss=0

    print("Epoch", epoch, "Completed")   

    show_cuda()
torch.cuda.empty_cache()
import gc

gc.collect()
output=[]

names = []

with torch.no_grad():

    for i, data in enumerate(testloader, 0):

        images, name = data    

        inputs = images.type(torch.FloatTensor)

        inputs, label = Variable(inputs.to(device)), Variable(label.to(device))

        output.append(model(inputs).cpu().numpy())

        names.append(name[0])
ops = output
indices_op=[]

for i in range(len(ops)):

    indices_op.append(np.where(ops[i]>=0.383)[1])

indices_op_str=[]

for i in range(len(indices_op)):

    indices_op_str.append(' '.join(map(str, indices_op[i])))
!mkdir ../models

torch.save(model.state_dict(), '../models/model1.pth')
d = {'id':names, 'attribute_ids':indices_op_str}

df = pd.DataFrame(d)
df
df.to_csv('submission.csv', index=False)