# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision

from torch import nn,optim

import torch.nn.functional as F

from torchvision import models,transforms,datasets

from torch.optim import lr_scheduler

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from torch.utils.data import *

import cv2

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import cohen_kappa_score

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



use_gpu = torch.cuda.is_available()

if use_gpu:

    print("Using CUDA")

    

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
basedirpath='../input/aptos2019-blindness-detection/'

traindata=pd.read_csv(basedirpath+'train.csv')

testdata=pd.read_csv(basedirpath+'test.csv')
print("Length of the training data",len(traindata))

traindata.head()
print(traindata['diagnosis'].value_counts())

print("Get all the training images")

trainimages=traindata['id_code'].values

trainy=traindata['diagnosis']
sampleimg=mpimg.imread('../input/aptos2019-blindness-detection/train_images/{}.png'.format(trainimages[4]))

print("Size of image",sampleimg.shape)

plt.imshow(sampleimg)

print("Label for the image is:",traindata[traindata['id_code']==re.split('/|\.','../input/train_images/{}.png'.format(trainimages[1]))[5]]['diagnosis'].values)

diagnosis = {0:'No DR', 

1:'Mild', 

2:'Moderate', 

3:'Severe', 

4:'Proliferative DR'}
dia = pd.DataFrame(traindata['diagnosis'].map(diagnosis))

dia.rename(columns={'diagnosis': "diagnosisText"},inplace=True)

traindata = pd.concat([traindata,dia],1)

traindata.head()
print("Preparing the y labels")



def prepare_labels(values):

    onehot_encoder = OneHotEncoder(sparse=False)

    values = values.reshape(len(values), 1)

    onehot_encoded = onehot_encoder.fit_transform(values)

    return onehot_encoded

trainy=prepare_labels(trainy.values)

print("Label encoding",trainy)
# Image transformations



image_transforms = {

    # Train uses data augmentation

    'train':

    transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(size=224),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

    ]),



    

    #Test

    'test':

    transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(size=224),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}
class AptosDataset(Dataset):



    def __init__(self, dataframe, root_dir,trainy,datatype, transform=None):

        

        self.dataframe = dataframe

        self.transform = transform

        self.datatype=datatype

        

        if self.datatype=='train':

            self.rootdir = basedirpath+'train_images/'

            self.labels=trainy

        else:

            self.rootdir = basedirpath+'test_images/'

            self.labels = np.zeros((self.dataframe.shape[0], 5))

        

    def __len__(self):

        return len(self.dataframe)



    def __getitem__(self, idx):

        img_name = os.path.join(self.rootdir,

                                self.dataframe.iloc[idx, 0])

        img = cv2.imread(img_name+'.png')

       

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image=self.transform(img)

        label=self.labels[idx]

        

        if self.datatype=='test':

            return image,label,img_name

        

        return image,label
def split_image_data(train_data,test_data,

                     batch_size=64,

                     num_workers=8,

                     valid_size=0.1,

                     sampler=SubsetRandomSampler):

    

    num_train = len(train_data)



    indices = list(range(num_train))

    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))



    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler(train_idx)

    valid_sampler = sampler(valid_idx)



    train_loader = torch.utils.data.DataLoader(train_data,

                                               batch_size=batch_size,

                                               sampler=train_sampler,

                                               num_workers=num_workers)



    valid_loader = torch.utils.data.DataLoader(train_data,

                                               batch_size=batch_size,

                                               sampler=valid_sampler,

                                               num_workers=num_workers)

    

    test_loader = torch.utils.data.DataLoader(test_data,

                                               batch_size=batch_size,

                                               num_workers=num_workers)



    return train_loader,valid_loader,test_loader
train=AptosDataset(traindata,basedirpath,trainy,'train',transform=image_transforms['train'])

test=AptosDataset(testdata,basedirpath,None,'test',transform=image_transforms['test'])



trainset,validset,testset=split_image_data(train,test)
images_batch, labels_batch = iter(trainset).next()

print(images_batch.shape)

print(labels_batch.shape)
model = torchvision.models.resnet50()

model.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))
model
num_ftrs = model.fc.in_features

num_ftrs
count = 0

for child in model.children():

    count+=1

print(count)
count = 0

for child in model.children():

  count+=1

  if count < 7:

    for param in child.parameters():

        param.requires_grad = False
model.fc = nn.Linear(num_ftrs, 5)
model
model.cuda()
criterion = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 1e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
def cohen_k_score(y_true , y_pred):

    skl = cohen_kappa_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), weights='quadratic')

    return skl
valid_ck_min = 0

patience = 10



# current number of epochs, where validation loss didn't increase

p = 0



# whether training should be stopped

stop = False



# number of epochs to train the model

n_epochs = 15

for epoch in range(1, n_epochs+1):

    print('Epoch:', format(epoch))



    train_loss = []

    train_ck_score = []



    for batch_i, (data, target) in enumerate(trainset):

        

        model.train()



        data, target = data.cuda(), target.cuda()



        optimizer.zero_grad()

        output = model(data)

        

        loss = criterion(output, target.float())

        train_loss.append(loss.item())

        

        a = target.data.cpu().numpy()

        b = output.detach().cpu().numpy()

        train_ck_score.append(cohen_k_score(a, b))

        

        loss.backward()

        optimizer.step()

        

    model.eval()

    val_loss = []

    val_ck_score = []

    for batch_i, (data, target) in enumerate(validset):

        data, target = data.cuda(), target.cuda()

        output = model(data)



        loss = criterion(output,target.float())



        val_loss.append(loss.item()) 

        

        a = target.data.cpu().numpy()

        b = output.detach().cpu().numpy()

        

        val_ck_score.append(cohen_k_score(a, b))



    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')

    print(f'Epoch {epoch}, train cohen: {np.mean(train_ck_score):.4f}, valid cohen: {np.mean(val_ck_score):.4f}.')

    



    val_ck_score = np.mean(val_ck_score)

    

    scheduler.step()

    

    if val_ck_score > valid_ck_min:

        print('Validation CK score increased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_ck_min,

        val_ck_score))

        valid_ck_min = val_ck_score

        p = 0



    # check if validation loss didn't improve

    if val_ck_score < valid_ck_min:

        p += 1

        print(f'{p} epochs of decreasing val ck score')

        if p > patience:

            print('Stopping training : Early Stopping')

            stop = True

            break        

            

    if stop:

        break
torch.save(model.state_dict(), 'modelresnet50.pt')
sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')



model.eval()

for (data, target, name) in testset:

    data = data.cuda()

    output = model(data)

    output = output.cpu().detach().numpy()

    for i, (e, n) in enumerate(list(zip(output, name))):

        sub.loc[sub['id_code'] == n.split('/')[-1].split('.')[0], 'diagnosis'] = np.argmax(e)

        

sub.to_csv('submission.csv', index=False)
sub.head()