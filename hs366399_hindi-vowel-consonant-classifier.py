import os

import random

import base64

import torch

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from PIL import Image

from torch.utils.data import DataLoader, Dataset, random_split



device = torch.device('cuda:0')

train_path = "../input/train/train/"

test_path = "../input/test/test/"

print(torch.cuda.get_device_name(0))
class HindiDataset(Dataset):

    

    def __init__(self, train_img_path, test_img_path, transform = None, train = True):

        self.train_img_path = train_img_path

        self.test_img_path = test_img_path

        self.train_img_files = os.listdir(train_img_path)

        self.test_img_files = os.listdir(test_img_path)

        self.transform = transform

        self.train = train

    

    def __len__(self):

        return len(self.train_img_files)

    

    def __getitem__(self, indx):

            

        if self.train:  

            

            if indx >= len(self.train_img_files):

                raise Exception("Index should be less than {}".format(len(self.img_files)))

               

            image = Image.open(self.train_img_path + self.train_img_files[indx]).convert('RGB')

            labels = self.train_img_files[indx].split('_')

            V = int(labels[0][1])

            C = int(labels[1][1])

            label = {'Vowel' : V, 'Consonant' : C}



            if self.transform:

                image = self.transform(image)



            return image, label

        

        if self.train == False:

            image = Image.open(self.test_img_path + self.test_img_files[indx]).convert('RGB')

            if self.transform:

                image = self.transform(image)



            return image, self.test_img_files[indx]
class BasicBlock(nn.Module):

    def __init__(self, channels = 256, stride = 1, padding = 1):

        super(BasicBlock, self).__init__()

        self.channels = channels

        self.stride = stride

        self.padding = padding

        

        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,

                                kernel_size = 3, stride = self.stride, padding = self.padding)

        self.bn_1 = nn.BatchNorm2d(self.channels)

        self.prelu_1 = nn.PReLU()

        

        self.conv_2 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,

                                kernel_size = 3, stride = self.stride, padding = self.padding)

        self.bn_2 = nn.BatchNorm2d(self.channels)

        self.prelu_2 = nn.PReLU()

        

        self.conv_3 = nn.Conv2d(in_channels = self.channels, out_channels = self.channels,

                                kernel_size = 5, stride = self.stride, padding = self.padding + 1)

        self.bn_3 = nn.BatchNorm2d(self.channels)

        

    def forward(self, x):

        identity = x

        x = self.prelu_1(self.bn_1(self.conv_1(x)))

        x = self.bn_2(self.conv_2(x)) + self.bn_3(self.conv_3(identity))

        x = self.prelu_2(x)        

        return x
class ModInception(nn.Module):

    def __init__(self, channels = 256, stride = 1, padding = 1):

        super(ModInception, self).__init__()

        self.channels = channels

        self.stride = stride

        self.padding = padding

        

        self.conv_1 = nn.Conv2d(in_channels = self.channels, out_channels = 70, kernel_size = 1,

                                stride = self.stride, padding = 0)

        self.conv_2 = nn.Conv2d(in_channels = self.channels, out_channels = 60, kernel_size = 3,

                                stride = self.stride, padding = 1)

        self.conv_3 = nn.Conv2d(in_channels = self.channels, out_channels = 126,kernel_size = 5,

                                stride = self.stride, padding = 2)

        self.bn = nn.BatchNorm2d(self.channels)

        self.prelu = nn.PReLU() 

        

    def forward(self, x):

        x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x)], dim=1)

        x = self.prelu(self.bn(x))

        return x
class ResNet(nn.Module):

    def __init__(self, block, incp_block):

        super(ResNet, self).__init__()

        self.block = block

        self.incp_block = incp_block

        self.input_conv = nn.Sequential(

                       nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1),

                       nn.BatchNorm2d(64),

                       nn.PReLU(),

            

                       nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),

                       nn.BatchNorm2d(128),

                       nn.PReLU(),                



                       nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),

                       nn.BatchNorm2d(256),

                       nn.PReLU(),

                       )

        

        self.layer_64x64 = self.make_layers(3)

        self.layer_32x32 = self.make_layers(2)

        self.layer_16x16 = self.make_layers(2)

        self.layer_8x8 = self.make_layers(2)

        self.layer_4x4 = self.make_layers(2)

        

        self.downsample_conv_1 = nn.Sequential(

                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),

                                 nn.BatchNorm2d(256),

                                 nn.PReLU()

                                 )

        

        self.downsample_conv_2 = nn.Sequential(

                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),

                                 nn.BatchNorm2d(256),

                                 nn.PReLU()

                                 )

        self.downsample_conv_3 = nn.Sequential(

                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),

                                 nn.BatchNorm2d(256),

                                 nn.PReLU()

                                 )

        self.downsample_conv_4 = nn.Sequential(

                                 nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2),

                                 nn.BatchNorm2d(256),

                                 nn.PReLU()

                                 )

        self.downsample_conv_5 = nn.Sequential(

                                 nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2),

                                 nn.BatchNorm2d(128),

                                 nn.PReLU()

                                 )

            

        self.V_classifier = nn.Sequential(

                    nn.Linear(512, 256),

                    nn.BatchNorm1d(256),

                    nn.PReLU(),

            

                    nn.Linear(256, 128),

                    nn.BatchNorm1d(128),

                    nn.PReLU(),

            

                    nn.Linear(128, 64),

                    nn.BatchNorm1d(64),

                    nn.PReLU(),

                

                    nn.Linear(64, 32),

                    nn.BatchNorm1d(32),

                    nn.PReLU(),

            

                    nn.Linear(32, 16),

                    nn.BatchNorm1d(16),

                    nn.PReLU(),

            

                    nn.Linear(16, 10)

                    )

        

        self.C_classifier = nn.Sequential(

                    nn.Linear(512, 256),

                    nn.BatchNorm1d(256),

                    nn.PReLU(),

            

                    nn.Linear(256, 128),

                    nn.BatchNorm1d(128),

                    nn.PReLU(),

            

                    nn.Linear(128, 64),

                    nn.BatchNorm1d(64),

                    nn.PReLU(),

                

                    nn.Linear(64, 32),

                    nn.BatchNorm1d(32),

                    nn.PReLU(),

            

                    nn.Linear(32, 16),

                    nn.BatchNorm1d(16),

                    nn.PReLU(),

            

                    nn.Linear(16, 10)

                    )

    

    def make_layers(self, layers):

        res_layers = []

        for i in range(layers):

            res_layers.append(self.block())

            res_layers.append(self.incp_block())

        return nn.Sequential(*res_layers)        

    

    def forward(self, x):

        

        x = self.input_conv(x)

        x = self.layer_64x64(x)

        x = self.downsample_conv_1(x)

        x = self.layer_32x32(x)

        x = self.downsample_conv_2(x)

        x = self.layer_16x16(x)

        x = self.downsample_conv_3(x)

        x = self.layer_8x8(x)

        x = self.downsample_conv_4(x)

        x = self.layer_4x4(x)

        x = self.downsample_conv_5(x)

        x = x.view(x.shape[0], -1)

        out_1 = self.V_classifier(x)

        out_2 = self.C_classifier(x)

        return out_1, out_2   
data = HindiDataset(train_path, test_path, transform = transforms.Compose([transforms.ToTensor(),

                                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

                                                       ]),

                                                       train = True)

train_size = int(0.9 * len(data))

test_size = len(data) - train_size



train_data, validation_data = random_split(data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)
deepNet = ResNet(BasicBlock, ModInception).to(device)

cnn_ce_loss = nn.CrossEntropyLoss()

cnnet_optim = optim.Adam(deepNet.parameters(), lr = 0.0002, weight_decay=0)
def get_accuracy(output, label, batch_size):

    

    label = label.detach().cpu().numpy().squeeze()

    output = output.detach().cpu()

    _, indices = torch.max(output, dim=1)

    output = torch.zeros_like(output)

    itr = iter(indices)

    for i in range(output.shape[0]):

        output[i, int(next(itr))] = 1

    

    label = torch.tensor(np.eye(10)[label]).float()

    diff = torch.sum(torch.abs(output - label))/(2*output.shape[0])

    acc = 100 - (100 * diff)

    

    return acc
epochs = 85

Costs = []

Accuracy = []



deepNet = deepNet.train()



for epoch in range(epochs):

    acc = 0

    count = 0

    for i, batch in enumerate(train_loader):

        

        count += 1

        images, label = batch

        images = images.to(device)

        label['Vowel'] = label['Vowel'].to(device).long()

        label['Consonant'] = label['Consonant'].to(device).long()

        

        cnnet_optim.zero_grad()

        out_1, out_2 = deepNet(images)

        loss_1 = cnn_ce_loss(out_1, label['Vowel'])

        loss_2 = cnn_ce_loss(out_2, label['Consonant'])

        loss = loss_1 + loss_2

        loss.backward()

        cnnet_optim.step()

        cost = loss.item()

        

        acc1 = get_accuracy(out_1, label['Vowel'], images.shape[0])

        acc2 = get_accuracy(out_2, label['Consonant'], images.shape[0])

        acc = acc + (acc1 + acc2)/2



    Accuracy.append(acc/(10*count))

    Costs.append(cost)    

    print("Epoch [{}/{}], Loss : {}, Accuracy : {}, acc_1 : {}, acc_2 : {}".format((epoch+1), epochs,

                                                                                   round(float(cost), 2),

                                                                                   round(float(Accuracy[-1].cpu()), 2),

                                                                                   round(float(acc1.cpu()), 2),

                                                                                   round(float(acc2.cpu()), 2)))



plt.title("Loss and Accuracy with iterations")

plt.plot(Costs, label = 'Cost')

plt.plot(Accuracy, label = 'Accuracy')

plt.xlabel("Iterations")

plt.ylabel("Loss and Accuracy")

plt.legend()

plt.show()



count = 0

acc = 0



deepNet = deepNet.eval()



for i, batch in enumerate(train_loader):

    

    count += 1

    images, label = batch

    images = images.to(device)

    label['Vowel'] = label['Vowel'].to(device).long()

    label['Consonant'] = label['Consonant'].to(device).long()



    out_1, out_2 = deepNet(images)

    out_1, out_2 = F.log_softmax(out_1, dim = 1), F.log_softmax(out_2, dim = 1)

    

    acc1 = get_accuracy(out_1, label['Vowel'], images.shape[0])

    acc2 = get_accuracy(out_2, label['Consonant'], images.shape[0])

    acc = acc + (acc1 + acc2)/2



print("Train Accuracy : {}".format(acc/count))
deepNet = deepNet.eval()



count = 0

acc = 0

for i, batch in enumerate(validation_loader):



    count += 1

    images, label = batch

    images = images.to(device)

    label['Vowel'] = label['Vowel'].to(device).long()

    label['Consonant'] = label['Consonant'].to(device).long()



    out_1, out_2 = deepNet(images)

    out_1, out_2 = F.log_softmax(out_1, dim = 1), F.log_softmax(out_2, dim = 1)

    

    acc1 = get_accuracy(out_1, label['Vowel'], images.shape[0])

    acc2 = get_accuracy(out_2, label['Consonant'], images.shape[0])

    acc = acc + (acc1 + acc2)/2



print("Test Accuracy : {}".format(acc/count))

test_data = HindiDataset(train_path, test_path, transform = transforms.Compose([transforms.ToTensor(),

                                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

                                                       train = False)

testset = DataLoader(test_data, batch_size = 32, shuffle = False)
preds = {}

img_ids = []

F_results = []



deepNet = deepNet.eval()



for i, batch in enumerate(testset):

    

    images, img_names = batch

    images = images.to(device)

    out_1, out_2 = deepNet(images)

    out_1, out_2 = F.log_softmax(out_1, dim = 1), F.log_softmax(out_2, dim = 1)

    out_1, out_2 = torch.max(out_1, dim=1)[1].cpu(), torch.max(out_2, dim=1)[1].cpu()

    

    for names,V,C in zip(img_names, out_1, out_2):

        F_results.append('V' + str(int(V)) + '_C' + str(int(C)))

        img_ids.append(names)

        

    if i%50 == 0:

        print("{} images tested....".format((i + 1) * images.shape[0]))
submission = pd.DataFrame({'ImageId':img_ids, 'Class':F_results})

submission = submission.sort_values(['ImageId'])

submission.to_csv("submission.csv", index=False)

submission.tail()