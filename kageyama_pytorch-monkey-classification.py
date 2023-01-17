# Import Modules

import glob

import os.path as osp

import random

import numpy as np

import json

from PIL import Image

from tqdm import tqdm

import matplotlib.pyplot as plt



# torch

import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

import torchvision

from torchvision import models,transforms
# set random seed

torch.manual_seed(1234)

np.random.seed(1234)

random.seed(1234)
# define label to name & name to id

label2name = {

    'n0':'alouatta_palliata',

    'n1':'erythrocebus_patas',

    'n2':'cacajao_calvus',

    'n3':'macaca_fuscata',

    'n4':'cebuella_pygmea',

    'n5':'cebus_capucinus',

    'n6':'mico_argentatus',

    'n7':'saimiri_sciureus',

    'n8':'aotus_nigriceps',

    'n9':'trachypithecus_johnii',

}

name2id = {

    'alouatta_palliata':0,

    'erythrocebus_patas':1,

    'cacajao_calvus':2,

    'macaca_fuscata':3,

    'cebuella_pygmea':4,

    'cebus_capucinus':5,

    'mico_argentatus':6,

    'saimiri_sciureus':7,

    'aotus_nigriceps':8,

    'trachypithecus_johnii':9,

}
class ImageTransform():

    '''

    This is image transform class. This class's action differs depending on the 'train' or 'val'. 

    It resize image size and normarize image color.

    Attributes

    -----------

    resize:int

        img size after resize



    mean : (R,G,B)

        average of each channel

    

    std : (R,G,B)

        standard deviation of each channel

    '''

    def __init__(self,resize,mean,std):

        self.data_transform = {

            'train':transforms.Compose([

                transforms.RandomResizedCrop(resize,scale=(0.5,1.0)),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(mean,std)

            ]),

            'val':transforms.Compose([

                transforms.Resize(resize),

                transforms.CenterCrop(resize),

                transforms.ToTensor(),

                transforms.Normalize(mean,std)

            ])

        }



    def __call__(self,img,phase='train'):

        return self.data_transform[phase](img)
# make data path list

def make_datapath_list(phase):

    rootpath = '/kaggle/input/10-monkey-species/'

    target_path = osp.join(rootpath+phase+'/**/**/*.jpg')

    path_list = []

    for path in glob.glob(target_path):

        path_list.append(path)    

    return path_list
train_list = make_datapath_list(phase='training')

val_list = make_datapath_list(phase='validation')
class Dataset(data.Dataset):

    '''

    Attributes

    -------------

    file_list:list

        data path list

    transform: object

        ImageTransform object

    phase: 

        'train' or 'val'

    '''

    def __init__(self,file_list,transform=None,phase='train'):

        self.file_list = file_list

        self.transform = transform

        self.phase = phase

    

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self,index):

        '''

        get after preprocessing image tensor and label 

        '''

        # 

        img_path = self.file_list[index]

        img = Image.open(img_path) 



        # preprocessing

        img_transformed = self.transform(img,self.phase) # torch.Size([3,224,224])



        # get image label from file name

        arr = img_path.split('/')

        label = arr[-2]

        name = label2name[label]



        # transform label to number

        label_num = name2id[name]



        return img_transformed,label_num
size = 224



# when we use pretrain models, we should normarize image following mean value and std value.

# reference : https://pytorch.org/docs/master/torchvision/models.html

mean = (0.485,0.456,0.406)

std = (0.229,0.224,0.225)



# make train dataset and val dataset

train_dataset = Dataset(file_list=train_list,transform=ImageTransform(size,mean,std),phase = 'train') 

val_dataset = Dataset(file_list=val_list,transform=ImageTransform(size,mean,std),phase = 'val')
# check move

index = 0

print(train_dataset.__getitem__(index)[0].size())

print(train_dataset.__getitem__(index)[1])
# define mini batch size

batch_size = 32



# make dataloader

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)



# put in dict 

dataloaders_dict = {'train':train_dataloader,'val':val_dataloader}
# check move

batch_iterator = iter(dataloaders_dict['train']) # convert iterator

inputs,labels = next(batch_iterator) # extract first items

print(inputs.size())

print(labels)
# load pretrain VGG-16 model

use_pretrained = True # use pretrain parameter

net = models.vgg16(pretrained=use_pretrained)



# final output layer 

net.classifier[6] = nn.Linear(in_features=4096,out_features=10)
# set train mode

net.train()



# set loss

criterion = nn.CrossEntropyLoss()



# fine tuning 

param_to_update_1 = []

param_to_update_2 = []

param_to_update_3 = []



# learn parameter list

update_param_names_1 = ['features']

update_param_names_2 = ['classifier.0.weight','classifier.0.bias','classifier.3.weight','classifier.3.bias']

update_param_names_3 = ['classifier.6.weight','classifier.6.bias']

# fix parameter

for name,param in net.named_parameters():

    print(name)

    if update_param_names_1[0] in name:

        param.requires_grad = True

        param_to_update_1.append(param)

    elif name in update_param_names_2:

        param.requires_grad = True

        param_to_update_2.append(param)

    elif name in update_param_names_3:

        param.requires_grad = True

        param_to_update_3.append(param)

    else:

        param.requires_grad = False
# set optimizer

optimizer = optim.SGD([

    {'params':param_to_update_1,'lr':1e-4},

    {'params':param_to_update_2,'lr':5e-4},

    {'params':param_to_update_3,'lr':1e-3},

    ],momentum=0.9)
def train_model(net,dataloaders_dict,criterion,optimizer,num_epochs):

    # init

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('use device:',device)

    net.to(device)

    

    # 

    history_train_loss = []

    history_train_acc = []

    history_val_loss = []

    history_val_acc = []



    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1,num_epochs))

        print('-------------------------------------')



        for phase in ['train','val']:

            if phase == 'train':

                net.train()

            else:

                net.eval()

            

            epoch_loss = 0.0

            epoch_corrects = 0



            # pick mini batch from dataloader

            for inputs,labels in tqdm(dataloaders_dict[phase]):

                inputs = inputs.to(device)

                labels = labels.to(device)

                # init optimizer

                optimizer.zero_grad()

                # calculate forward

                with torch.set_grad_enabled(phase=='train'):

                    outputs = net(inputs)

                    loss = criterion(outputs,labels) # calculate loss

                    _,preds = torch.max(outputs,1) # predict label



                    # backward (train only)

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                    

                    # update loss sum

                    epoch_loss += loss.item() * inputs.size(0)

                    # correct answer count 

                    epoch_corrects += torch.sum(preds == labels.data)

            # show loss and correct answer rate per epoch 

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            if phase == 'train':

                history_train_loss.append(epoch_loss)

                history_train_acc.append(epoch_acc)

            else:

                history_val_loss.append(epoch_loss)

                history_val_acc.append(epoch_acc)

    return history_train_loss,history_train_acc,history_val_loss,history_val_acc

num_epochs=10

train_loss,train_acc,val_loss,val_acc = train_model(net,dataloaders_dict,criterion,optimizer,num_epochs=num_epochs)
import matplotlib.pyplot as plt
plt.plot(train_loss)

plt.plot(val_loss)

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'val'])

plt.show()
plt.plot(train_acc)

plt.plot(val_acc)

plt.title('Model Acc')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'val'])

plt.show()
save_path = './model.pth'

torch.save(net.state_dict(),save_path)
load_path = './model.pth'

load_weights = torch.load(load_path)

net.load_state_dict(load_weights)
from sklearn.metrics import classification_report



pred = []

Y = []

X = []

for i, (x,y) in enumerate(val_dataloader):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)

    x = x.to(device)

    y = y.to(device)

    with torch.no_grad():

        output = net(x)

    pred += [int(l.argmax()) for l in output]

    Y += [int(l) for l in y]

    X += [l.cpu() for l in x]



print(classification_report(Y, pred,target_names=[v for v in label2name.values()]))
i=0

prop_class=[]

mis_class=[]



for i in range(len(Y)):

    if(Y[i]==pred[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(Y)):

    if(Y[i]!=pred[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(X[prop_class[count]].numpy().transpose(1,2,0))

        ax[i,j].set_title("Predicted : "+str(label2name['n{}'.format(pred[prop_class[count]])])+"\n"+"Actual : "+str(label2name['n{}'.format(pred[prop_class[count]])]))

        plt.tight_layout()

        count+=1