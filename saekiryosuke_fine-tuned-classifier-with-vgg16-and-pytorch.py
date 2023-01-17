# import packages

import glob

import os.path as osp

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
# setting random number seed. Arbitrary seed is OK.

torch.manual_seed(1234)

np.random.seed(1234)

random.seed(1234)



# preprocess class for each image

class ImageTransform():



    def __init__(self, resize, mean, std):

        self.data_transform = {

            'train': transforms.Compose([

                # data augmentation

                transforms.RandomResizedCrop(

                   resize, scale=(0.5, 1.0)),

                transforms.RandomHorizontalFlip(), 

                # convert to tensor for PyTorch

                transforms.ToTensor(),

                # color normalization

                transforms.Normalize(mean, std)

            ]),

            'val': transforms.Compose([

                transforms.CenterCrop(resize),

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ])

        }



    def __call__(self, img, phase='train'):



        return self.data_transform[phase](img)



# Showing one result of the preprocess



image_file_path = '../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_479.jpeg'



img_originalsize = Image.open(image_file_path)   # [height][width][color RGB]

img = img_originalsize.resize((256, 256))



img = img.convert("L").convert("RGB")



# original image

plt.imshow(img)

plt.show()



# after preprocess

size = 256

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)



transform = ImageTransform(size, mean, std)

img_transformed = transform(img, phase="train")  # torch.Size([3, 224, 224])



# (color, height, width) -> (height, width, color), normalize colors in the range (0 - 1)

img_transformed = img_transformed.numpy().transpose((1, 2, 0))

img_transformed = np.clip(img_transformed, 0, 1)

plt.imshow(img_transformed)

plt.show()

# making file path list

def make_datapath_list(phase="train"):

    

    rootpath = "../input/chest-xray-pneumonia/chest_xray/"

    

    target_path = osp.join(rootpath+phase+'/**/*.jpeg')

    print(target_path)



    path_list = []



    # getting file paths

    for path in glob.glob(target_path):

        path_list.append(path)



    return path_list





train_list = make_datapath_list(phase="train")

val_list = make_datapath_list(phase="val")
# making dataset



class lungDataset(data.Dataset):



    def __init__(self, file_list, transform=None, phase='train'):

        self.file_list = file_list

        self.transform = transform

        self.phase = phase



    def __len__(self):

        return len(self.file_list)



    def __getitem__(self, index):



        # load image

        img_path = self.file_list[index]

        

        img_originalsize = Image.open(img_path)

        # resize

        img = img_originalsize.resize((256, 256))

        

        # grey -> color

        img = img.convert("L").convert("RGB")



        # preprocess

        img_transformed = self.transform(

            img, self.phase)  # torch.Size([3, 224, 224])



        # picking up labels

        if self.phase == "train":

            label = img_path[47:53]



        elif self.phase == "val":

            label = img_path[45:51]



        # label char -> number

        if label == "NORMAL":

            label = 0



        elif label == "PNEUMO":

            label = 1



        return img_transformed, label





# run

train_dataset = lungDataset(

    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')



val_dataset = lungDataset(

    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')



# motion check

# index = 0

# print(train_dataset.__getitem__(index)[0].size())

# print(train_dataset.__getitem__(index)[1])



# print(val_dataset.__getitem__(index)[0].size())

# print(val_dataset.__getitem__(index)[1])

batch_size = 32



# making dataloader

train_dataloader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=True)



val_dataloader = torch.utils.data.DataLoader(

    val_dataset, batch_size=batch_size, shuffle=False)



# put dataloader into dictionary type

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}



# motion check

# batch_iterator = iter(dataloaders_dict["train"])



# inputs, labels = next(

#     batch_iterator)  # pick up first element

# print(inputs.size())

# print(labels)

# load pretrained vgg16 from PyTorch as an instance

# need to make setting 'internet' to 'On'.

use_pretrained = True

net = models.vgg16(pretrained=use_pretrained)



# Replace output layer for 2 class classifier, 'NORMAL' and 'PNEUMONIA'.

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)



net.train()
# setting of loss function

criterion = nn.CrossEntropyLoss()



# setting fine tuned parameters



params_to_update_1 = []

params_to_update_2 = []

params_to_update_3 = []



# Not only output layer, "features" layers and other classifier layers are tuned.

update_param_names_1 = ["features"]

update_param_names_2 = ["classifier.0.weight",

                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]

update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]



# store parameters in list

for name, param in net.named_parameters():

    if update_param_names_1[0] in name:

        param.requires_grad = True

        params_to_update_1.append(param)

        #print("params_to_update_1:", name)



    elif name in update_param_names_2:

        param.requires_grad = True

        params_to_update_2.append(param)

        #print("params_to_update_2:", name)



    elif name in update_param_names_3:

        param.requires_grad = True

        params_to_update_3.append(param)

        #print("params_to_update_3:", name)



    else:

        param.requires_grad = False

        #print("no learning", name)



# print("-----------")

# print(params_to_update_1)



# Learning Rates

optimizer = optim.SGD([

    {'params': params_to_update_1, 'lr': 1e-4},

    {'params': params_to_update_2, 'lr': 5e-4},

    {'params': params_to_update_3, 'lr': 1e-3}

], momentum=0.9)

# training function

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    

    accuracy_list = []

    loss_list = []

    

    # Precondition : Accelerator GPU -> 'On'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using device：", device)



    # put betwork into GPU

    net.to(device)

    torch.backends.cudnn.benchmark = True



    # epoch loop

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-------------')



        for phase in ['train', 'val']:

            if phase == 'train':

                net.train()  # set network 'train' mode

            else:

                net.eval()   # set network 'val' mode



            epoch_loss = 0.0

            epoch_corrects = 0



            # Before training

            if (epoch == 0) and (phase == 'train'):

                continue

            

                      

            # batch loop

            for inputs, labels in tqdm(dataloaders_dict[phase]):

                   

                # send data to GPU

                inputs = inputs.to(device)

                labels = labels.to(device)

                

                # initialize optimizer

                optimizer.zero_grad()



                # forward

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(inputs)



                    loss = criterion(outputs, labels)  #calcurate loss

                    _, preds = torch.max(outputs, 1)  # predict

  

                    # back propagtion

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                    # update loss summation

                    epoch_loss += loss.item() * inputs.size(0)  

                    # update correct prediction summation

                    epoch_corrects += torch.sum(preds == labels.data)



            # loss and accuracy for each epoch loop

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

            epoch_acc = epoch_corrects.double(

            ) / len(dataloaders_dict[phase].dataset)

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

            

            if phase == 'val':

                accuracy_list.append(epoch_acc.item())

                loss_list.append(epoch_loss)

            

    return accuracy_list, loss_list

# start training

num_epochs=10

accuracy_list, loss_list = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
# To save trained model

# save_path = './weights_fine_tuning.pth'

# torch.save(net.state_dict(), save_path)

epoch_num = list(range(10))

fig, ax = plt.subplots(facecolor="w")

ax.plot(epoch_num, accuracy_list, label="accuracy")

ax.plot(epoch_num, loss_list, label="loss")

plt.xticks(epoch_num) 



ax.legend()



plt.show()