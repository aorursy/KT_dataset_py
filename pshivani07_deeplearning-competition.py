import time

import copy



import matplotlib.pyplot as plt

import numpy as np

import PIL

import math

from PIL import Image



import torch

from torch import nn, optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split

import os



from sklearn import preprocessing
train_on_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
le = preprocessing.LabelEncoder()

train['Class'] = le.fit_transform(train['Class'])

train.head()
transform_train = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

transform_test = transforms.Compose([

        transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], 

                            [0.229, 0.224, 0.225])

    ])
class ImageClassifierDataset(Dataset):

    def __init__(self, img_root, lab_df, training=True, transform=None):

        self.img_root = img_root

        self.transform = transform

        self.lab_df = lab_df

        self.train = training

        self.file_names = [file for _,_,files in os.walk(self.img_root) for file in files]

        

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name = self.file_names[index]

        img_data = self.image_data(self.img_root+"/"+file_name)

        if self.transform:

            img_data = self.transform(img_data)

        if self.train:

            ind = self.label_data(file_name)

            label = torch.zeros(4)

            label[ind] = 1

            return img_data, label

        else:

            return img_data, file_name

    

    def image_data(self, path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            img = img.resize((224,224), Image.ANTIALIAS)

            return img.convert('RGB')

        

    def label_data(self, img_name):

        label = self.lab_df.loc[self.lab_df['Image'] == img_name]

        label = label['Class']

        return int(label)
full_data = ImageClassifierDataset("../input/Train Images/Train Images",train,training=True, transform=transform_train)

train_size = int(0.8 * len(full_data))

test_size = len(full_data) - train_size



batch_size = 64



train_data, validation_data = random_split(full_data, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_data = ImageClassifierDataset("../input/Test Images/Test Images",test,training=False, transform=transform_test)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False)
vgg = models.resnet50(pretrained=True)
dataloader = {'train':train_loader, 'valid':validation_loader} 
for name, child in vgg.named_children():

    print(name)
for name, child in vgg.named_children():

    if name in ['layer3', 'layer4','avgpool']:

        print(name + ' is unfrozen')

        for param in child.parameters():

            param.requires_grad = True

    else:

        print(name + ' is frozen')

        for param in child.parameters():

            param.requires_grad = False
class Net(nn.Module):

    def __init__(self, model, p1, p2):

        super(Net, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Sequential(

            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),

            nn.Dropout(p1),

            nn.Linear(2048, 1000, bias = True),

            nn.ReLU(),

            nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1),

            nn.Dropout(p2),

            nn.Linear(1000, 4, bias = True)

        )

        

    def forward(self, x):

        out = self.features(x)

        out = out.view(out.size(0),-1)

        output = self.classifier(out)

        return output
def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):

    if scale_func == None:

        if mode == 'triangular':

            scale_fn = lambda x: 1.

            scale_mode = 'cycles'

        elif mode == 'triangular2':

            scale_fn = lambda x: 1 / (2.**(x - 1))

            scale_mode = 'cycles'

        elif mode == 'exp_range':

            scale_fn = lambda x: gamma**(x)

            scale_mode = 'iterations'

        else:

            raise ValueError(f'The {mode} is not valid value!')

    else:

        scale_fn = scale_func

        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    

    def rel_val(iteration, stepsize, mode):

        cycle = math.floor(1 + iteration / (2 * stepsize))

        x = abs(iteration / stepsize - 2 * cycle + 1)

        if mode == 'cycles':

            return max(0, (1 - x)) * scale_fn(cycle)

        elif mode == 'iterations':

            return max(0, (1 - x)) * scale_fn(iteration)

        else:

            raise ValueError(f'The {scale_mode} is not valid value!')

    return lr_lambda
def train_model(model, loss_fn, wtdecay, num_epochs=25):

    best_model_wts = model.state_dict()

    best_acc = 0.0

    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=0.01, amsgrad=True, weight_decay=wtdecay)

    clr = cyclical_lr(64, min_lr=0.0001, max_lr=0.1, mode='exp_range')

    scheduler = lr_scheduler.LambdaLR(opt, [clr])

  

    for epoch in range(num_epochs):

        print('Epoch %d/%d' % (epoch, num_epochs - 1))

        print('-'*20)

    

        for mode in ['train', 'valid']:

            if mode == 'train':

                model.train()

            else:

                model.eval()

        

            running_loss = 0.0

            running_correct = 0

            for data in dataloader[mode]:

                inputs, label = data

                inputs, label = Variable(inputs.to(device)), Variable(label.to(device))

                

                label = label.to(dtype=torch.long)

                

                _, label = torch.max(label.data, 1)

                

                opt.zero_grad()

        

                output = model(inputs)

            

                _, pred = torch.max(output.data, 1)

        

                loss = loss_fn(output, label)

        

                if mode == 'train':

                    loss.backward()

                    opt.step()

                    scheduler.step()

          

                running_loss += loss.item()

                running_correct += (torch.sum(pred == label.data))*1

        

            epoch_loss = running_loss/len(dataloader[mode])

            epoch_acc = running_correct/len(dataloader[mode])

      

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

      

            if mode == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = model.state_dict()

                

            del inputs, label, output, pred

            torch.cuda.empty_cache()

    print('Best val Acc : {:4f}'.format(best_acc))

    

    model.load_state_dict(best_model_wts)

  

    return model
loss_fn = nn.CrossEntropyLoss()

p1 = 0.3

wt = 0.01

net = Net(vgg, p1, p1)

net.to(device)

model = train_model(net, loss_fn, wt, num_epochs=45)
def predict(inputs, model):

    pred = model.forward(inputs)

    _, pred = torch.max(pred.data,1)

    return pred
def sample(model, dataloader):

    output = pd.DataFrame(columns=('Image','Class'))

    

    for i, (inputs, ind) in enumerate(dataloader):

        model.eval()

        inputs = inputs.to(device)

        pred = predict(inputs, model)

        classes = ['Attire','Decorationandsignage','Food','misc']

        output.loc[i] = [''.join(ind), classes[pred]]

        

    return output
submission = sample(model, test_loader)
submission.to_csv("submission.csv", index = False)