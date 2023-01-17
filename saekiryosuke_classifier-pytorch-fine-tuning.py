# import packages

import glob

import os.path as osp

import os

import random

import numpy as np

import json

from PIL import Image

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline



import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

import torchvision

from torchvision import models, transforms



import re

import csv



torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Setting random number seed.

torch.manual_seed(1234)

np.random.seed(1234)

random.seed(1234)
# Preprocess for images.

# When "training", it has data augumentaion.



class ImageTransform():

    """

    画像の前処理クラス。訓練時、検証時で異なる動作をする。

    画像のサイズをリサイズし、色を標準化する。



    Attributes

    ----------

    resize : int

        resize of image (image files have different size.)

    mean : (R, G, B)

        standardization colors

    std : (R, G, B)

        standardization colors

    """



    def __init__(self, resize, mean, std):

        self.data_transform = {

            'train': transforms.Compose([

                transforms.RandomResizedCrop(

                   resize, scale=(0.5, 1.0)),  # data augumentation

                transforms.RandomHorizontalFlip(),  # data augumentation

                transforms.ToTensor(),  # to Tensor

                transforms.Normalize(mean, std)  # standerdization

            ]),

            'val': transforms.Compose([

                # transforms.Resize(resize),  # resize

                transforms.CenterCrop(resize),  # picking up the center of the image resize×resize

                transforms.ToTensor(),  # to Tensor

                transforms.Normalize(mean, std)  # standardization

            ])

        }



    def __call__(self, img, phase='train'):

        """

        Parameters

        ----------

        phase : 'train' or 'val'

            setting mode

        """

        return self.data_transform[phase](img)

# check the preprocess



# 1. open image

image_file_path = '../input/training/train/cat.10005.jpg'



img_originalsize = Image.open(image_file_path)   # [height][width][RGB]

img = img_originalsize.resize((256, 256))



# 2. show original image

plt.imshow(img)

plt.show()



# 3. show image after preprocess

size = 256

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)



transform = ImageTransform(size, mean, std)

img_transformed = transform(img, phase="train")  # torch.Size([3, 224, 224])





img_transformed = img_transformed.numpy().transpose((1,2,0))

plt.imshow(img_transformed)

plt.show()

# making path list



def make_datapath_list(phase="train"):

    """

    

    Parameters

    ----------

    phase : 'train' or 'val'

        setting mode



    Returns

    -------

    path_list : list

        

    """



    rootpath = "../input/"

    

    if phase == 'train':

        target_path = osp.join(rootpath+'training/train/*.jpg')

    else :

        target_path = osp.join(rootpath+'valuate/val/*.jpg')

        

    path_list = []



    # getting file path

    for path in glob.glob(target_path):

        path_list.append(path)



    return path_list





# run

train_list = make_datapath_list(phase="train")

val_list = make_datapath_list(phase="val")
# making data set successing Pytorch Dataset class



class dogsAndCatsDataset(data.Dataset):

    

    def __init__(self, file_list, transform=None, phase='train'):

        self.file_list = file_list

        self.transform = transform

        self.phase = phase  # setting mode "train" or "test1"



    def __len__(self):

        

        return len(self.file_list)



    def __getitem__(self, index):

        

        # loading image for each index

        img_path = self.file_list[index]

        

        # open image file

        img_originalsize = Image.open(img_path)   # [height][width][RGB]

        img = img_originalsize.resize((256, 256))

        

        # preprocess of image

        img_transformed = self.transform(

            img, self.phase)  # torch.Size([3, 256, 256])

        

        # pick up label string from file name

        if self.phase == "train":

            label = img_path[24:27]

            

        elif self.phase == "val":

            label = img_path[21:24]

            

        # Convert label string -> number

        if label == "cat":

            label = 0

        

        elif label == "dog":

            label = 1



        return img_transformed, label





# making dataset

train_dataset = dogsAndCatsDataset(

    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')



val_dataset = dogsAndCatsDataset(

    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')



# check the motion

# index = 0

# print(train_dataset.__getitem__(index)[0].size())

# print(train_dataset.__getitem__(index)[1])



# print(val_dataset.__getitem__(index)[0].size())

# print(val_dataset.__getitem__(index)[1])

# setting butch size

batch_size = 32



# making data loader

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True)



val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False)



# into dictionary type

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}



# check the motion

# batch_iterator = iter(dataloaders_dict["train"])  # convert it to iterator



# inputs, labels = next(

#     batch_iterator)  # 1番目の要素を取り出す

# print(inputs.size())

# print(labels)
# loading vgg16 pretrained model

use_pretrained = True

net = models.vgg16(pretrained=use_pretrained)



# convert output layer for 2 class classifier

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)



# setting train mode

net.train()
# setting loss function

criterion = nn.CrossEntropyLoss()
# setting fine tuning parameters

params_to_update_1 = []

params_to_update_2 = []

params_to_update_3 = []



update_param_names_1 = ["features"]

update_param_names_2 = ["classifier.0.weight",

                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]

update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]



for name, param in net.named_parameters():

    if update_param_names_1[0] in name:

        param.requires_grad = True

        params_to_update_1.append(param)

        #print("params_to_update_1に格納：", name)



    elif name in update_param_names_2:

        param.requires_grad = True

        params_to_update_2.append(param)

        #print("params_to_update_2に格納：", name)



    elif name in update_param_names_3:

        param.requires_grad = True

        params_to_update_3.append(param)

        #print("params_to_update_3に格納：", name)



    else:

        param.requires_grad = False

        #print("勾配計算なし。学習しない：", name)

# I use SDG as optimizer.

optimizer = optim.SGD([

    {'params': params_to_update_1, 'lr': 1e-4},

    {'params': params_to_update_2, 'lr': 5e-4},

    {'params': params_to_update_3, 'lr': 1e-3}

], momentum=0.9)

# training function

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    

    train_accuracy_list = []

    train_loss_list = []

    

    valuate_accuracy_list = []

    valuate_loss_list = []

    

    # setting GPU

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #print("使用デバイス：", device)



    # network into GPU

    net.to(device)

    torch.backends.cudnn.benchmark = True



    # epoch loop

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-------------')



        # training and validation

        for phase in ['train', 'val']:

            if phase == 'train':

                net.train()

            else:

                net.eval()



            epoch_loss = 0.0

            epoch_corrects = 0



            # check accuracy before training

            if (epoch == 0) and (phase == 'train'):

                continue

            

                      

            # butch loop

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                   

                # send data GPU

                inputs = inputs.to(device)

                labels = labels.to(device)

                

                # initialize the optimizer

                optimizer.zero_grad()



                # forward propagation

                with torch.set_grad_enabled(phase == 'train'):

                    

                    outputs = net(inputs)

                                        

                    loss = criterion(outputs, labels) # output loss

                    _, preds = torch.max(outputs, 1)  # predict class

                    

                    # back propagation

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                    # sum of loss

                    epoch_loss += loss.item() * inputs.size(0)  

                    # sum of correct prediction

                    epoch_corrects += torch.sum(preds == labels.data)

            

            # loss and accuracy for each epoch loop

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_acc = epoch_corrects.double(

                ) / len(dataloaders_dict[phase].dataset)

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

            

            if phase == 'val':

                valuate_accuracy_list.append(epoch_acc.item())

                valuate_loss_list.append(epoch_loss)

            else:

                train_accuracy_list.append(epoch_acc.item())

                train_loss_list.append(epoch_loss)

        

    return train_accuracy_list, train_loss_list, valuate_accuracy_list, valuate_loss_list
# training 10 epochs (maybe too many...)

num_epochs=10

train_accuracy_list, train_loss_list, valuate_accuracy_list, valuate_loss_list = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
# plot training results



epoch_num = list(range(num_epochs-1))

fig, ax = plt.subplots(facecolor="w")

ax.plot(epoch_num, train_accuracy_list, label="train")

ax.plot(epoch_num, valuate_accuracy_list[1:], label="valuate")



plt.xticks(epoch_num) 



ax.legend()

fig = plt.title("accuracy")



plt.show()



fig, ax = plt.subplots(facecolor="w")



ax.plot(epoch_num, train_loss_list, label="train")

ax.plot(epoch_num, valuate_loss_list[1:], label="valuate")



plt.xticks(epoch_num) 



ax.legend()

fig = plt.title("loss")



plt.show()

# if save the model

# save_path = 'weights.pth'

# torch.save(net.state_dict(), save_path)





# start inferrence



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net.to(device)

net.eval()

    
# input test1 data

def make_test1_datapath_list():

    rootpath = "../input/testing1/"

    

    target_path = osp.join(rootpath+'test1/*.jpg')

    print(target_path)



    path_list = []



    # getting path

    for path in glob.glob(target_path):

        path_list.append(path)



    return path_list



# run

test1_list = make_test1_datapath_list()



ids = []

predictions = []

size = 256

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)



for path in test1_list:

    img_originalsize = Image.open(path)   # [height][width][RGB]

    img = img_originalsize.resize((256, 256))

    

    transform = ImageTransform(size, mean, std)

    img_transformed = transform(img, phase="val")  # torch.Size([3, 256, 256])

    

    img_for_net = img_transformed.unsqueeze(0)

    # into GPU

    img_for_net = img_for_net.to(device)

    outputs = net(img_for_net)

    

    # predict class

    _, preds = torch.max(outputs, 1)

    

    # print(re.split('[./]',path))

    splitted = re.split('[./]',path)

    test1_id = splitted[-2]

    

    ids.append(test1_id)

    predictions.append(preds.item())

# make submit csv

submitFormat = [ids, predictions]

submitFormat_t = np.array(submitFormat).T



print(submitFormat_t)



with open('submit.csv', 'w') as f:

    writer = csv.writer(f)

    

    fieldnames = ['id', 'label']

    #writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writerow(fieldnames)

    writer.writerows(submitFormat_t)

    