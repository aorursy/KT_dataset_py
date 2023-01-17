# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

import torch.nn.functional as F

from torchvision import datasets,transforms,models

from torch.utils.data import DataLoader,random_split,Dataset

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from shutil import copyfile

from PIL import Image

from torch.autograd import Variable

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

    print(os.path.join(dirname))



# Any results you write to the current directory are saved as output.
initial_directory='/kaggle/input/flowers-recognition/flowers'

data_directory='flower'

labels=[]

encoder=LabelEncoder()

encoder.fit(os.listdir(initial_directory))

if not os.path.exists(data_directory):os.makedirs(data_directory)

for item in os.listdir(initial_directory):

    if item!='flowers':

        for element in os.listdir(initial_directory+"/"+item):

            if '.py' not in element:

                if not os.path.exists(data_directory[0:]+'/'+element):

                    copyfile(initial_directory[0:]+'/'+item+'/'+element,data_directory[0:]+'/'+element)

                labels.append((element,encoder.transform([item]).tolist()[0])) 

df=pd.DataFrame(data=labels,columns=['id','labels']).sample(frac=1).reset_index(drop=True)
class FlowerDataset(Dataset):

    def __init__(self,root,list_id,labels,transform=None):

        self.root=root

        self.list_id=list_id

        self.labels=labels

        self.transform=transform

    def __len__(self):

        return len(self.list_id)

    def __getitem__(self,index):

        ID= self.list_id[index]

        

        X=Image.open(self.root+ID)

        

        if self.transform is not None:

             for transform_item in self.transform:

                X=transform_item(X)

        y=self.labels[index]

        return X,y
image_height,image_width=224,224

train_transform=[transforms.transforms.Resize(size=(image_height,image_width),interpolation=Image.BILINEAR),

                 transforms.transforms.RandomResizedCrop(image_height,scale=(0.8,0.8)),

                 transforms.transforms.RandomRotation(degrees=(-45,45)),

                 transforms.transforms.RandomHorizontalFlip(),

                 transforms.transforms.ColorJitter(brightness=0.4),

                 transforms.transforms.ToTensor(),

                 transforms.transforms.Normalize(mean=(0.4585,0.4196,0.3003),std=(0.2903,0.2592,0.2826))

                 ]

test_transform=[transforms.transforms.Resize(size=(image_height,image_width),interpolation=Image.BILINEAR),

                                         transforms.transforms.ToTensor(),

                                         transforms.transforms.Normalize(mean=(0.4585,0.4196,0.3003),std=(0.2903,0.2592,0.2826))

                                        ]



train_set,test_set=train_test_split(df,test_size=0.2)

train_data=FlowerDataset(root='flower/',list_id=train_set['id'].tolist(),labels=train_set['labels'].tolist(),transform=train_transform)

test_data=FlowerDataset(root='flower/',list_id=test_set['id'].tolist(),labels=test_set['labels'].tolist(),transform=test_transform)

num_epochs=25

batch_size=64

train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)

test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=4)
model=models.vgg16(pretrained=True)

torch.cuda.empty_cache() 

model.avgpool=nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),nn.Dropout(0.25,inplace=True))
fc=nn.Sequential(nn.Linear(in_features=512,out_features=len(encoder.classes_.tolist())))

model.classifier=fc

for param in model.classifier.parameters():

    param.requires_grad = True

for i,feat in enumerate(model.features.parameters()):

    if i<24:

        feat.requires_grad=False    

model.features        
if torch.cuda.is_available():

    model.cuda()

criterion=nn.CrossEntropyLoss()

lr=0.001

optimizer=torch.optim.Adam([{'params':model.classifier.parameters(),'lr':lr},

{ 'params': model.features[29].parameters(), 'lr': lr},                            

{ 'params': model.features[27].parameters(), 'lr': lr/10},

])

schedule=ReduceLROnPlateau(optimizer,mode='max',factor=0.6,patience=5,verbose=True)
def train_model(num_epoch):

    val_loss=0.0

    val_acc=0.0

    train_loss=0.0

    train_acc=0.0

    train_loss_list=[]

    test_loss_list=[]

    train_acc_list=[]

    test_acc_list=[]

    torch.cuda.empty_cache() 

    for i in range(num_epoch):

        schedule.step(val_acc)

        for image,label in train_loader:

            if torch.cuda.is_available():

                image=Variable(image.cuda())

                label=Variable(label.cuda())

            optimizer.zero_grad()    

            output=model(image)

            loss=criterion(output,label)

            loss.backward()

            optimizer.step()

            train_loss+=loss.data*image.size(0)

            _,prediction=torch.max(output.data,1)

            train_acc+=torch.sum(prediction.cpu()==label.data.cpu())       

        with torch.no_grad():

            model.eval()

            for image,label in test_loader:

                if torch.cuda.is_available():

                    image=Variable(image.cuda())

                    label=Variable(label.cuda())

                output=model(image)

                loss=criterion(output,label)

                _,prediction=torch.max(output.data,1)

                val_loss+=loss.data*image.size(0)

                val_acc+=torch.sum(prediction.cpu()==label.data.cpu())

        val_loss,val_acc=val_loss/(test_set.shape[0]),val_acc/(test_set.shape[0])    

        train_loss,train_acc=train_loss/(train_set.shape[0]),train_acc/(train_set.shape[0])

        train_loss_list.append(train_loss)

        test_loss_list.append(val_loss)

        train_acc_list.append(train_acc)

        test_acc_list.append(val_acc)

        print("Epoch : {} ,train_loss: {} ,test_loss: {} ,train_acc: {} ,test_acc: {} ".format(i,train_loss,val_loss,train_acc,val_acc))

    return train_loss_list,test_loss_list,train_acc_list,test_acc_list
if __name__=='__main__':

    train_loss_list,test_loss_list,train_acc_list,test_acc_list=train_model(num_epochs)
import matplotlib.pyplot as plt

images,labels=next(iter(test_loader))

with torch.no_grad():

    model.eval()

    index=15

    pred=model(images.cuda())

    plt.imshow(images[index].numpy().T)

    plt.show()

    print("Prediction:{}".format(encoder.inverse_transform(torch.argmax(pred.cpu(),axis=1))[index].item()))