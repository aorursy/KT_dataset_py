# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _,_ in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import torch.nn as nn

from torchvision import datasets, models, transforms

import torch.optim as optim

from torch.optim import lr_scheduler

from PIL import Image

import time

import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clean_path = '/kaggle/input/artificial-lunar-rocky-landscape-dataset/images/clean/'

ground_path = '/kaggle/input/artificial-lunar-rocky-landscape-dataset/images/ground/'

render_path = '/kaggle/input/artificial-lunar-rocky-landscape-dataset/images/render/'

real_path = '/kaggle/input/artificial-lunar-rocky-landscape-dataset/real_moon_images/'

clean_list = sorted([clean_path + i for i in os.listdir(clean_path)])

ground_list = sorted([ground_path + i for i in os.listdir(ground_path)])

render_list = sorted([render_path + i for i in os.listdir(render_path)])

real_list = sorted([real_path + i for i in os.listdir(real_path)])

print(len(clean_list), len(ground_list), len(render_list), len(real_list))
im_clean = Image.open(clean_list[0])

im_ground = Image.open(ground_list[0])

im_render = Image.open(render_list[0])

im_real = Image.open(real_list[0])

fig, axes = plt.subplots(2,2, figsize=(15,15))

print(axes.shape)

axes[0,0].imshow(im_clean)

axes[0,1].imshow(im_ground)

axes[1,0].imshow(im_render)

axes[1,1].imshow(im_real)
np.array(im_render).shape
class LunarDataset(object):

    def __init__(self, renders_list, seg_list):

        # load all image files, sorting them to

        # ensure that they are aligned

        self.moon_imgs = renders_list

        self.seg_imgs = seg_list

        

        assert len(renders_list) == len(seg_list)

        self.total_ims = len(renders_list)



    def __getitem__(self, idx):

        # load images

        img = Image.open(self.moon_imgs[idx])

        img = img.convert(mode='RGB')

        img = img.resize((720, 480))

        img = np.array(img)

        #img = transforms.functional.resize(img, (256,1600))

        #img = transforms.functional.to_tensor(img)

        img = img.reshape(3, 480, 720)

        img = torch.tensor(img)

        #img = img.view(3,256,1600)

        img = img.float()

        img = img/255

        

        mask = Image.open(self.seg_imgs[idx])

        #mask = mask.convert(mode='RGB')

        mask = mask.resize((720, 480))

        mask = np.array(mask)

        #img = transforms.functional.resize(img, (256,1600))

        #img = transforms.functional.to_tensor(img)

        mask = mask.reshape(480, 720,3)

        out_mask = np.zeros((mask.shape[0], mask.shape[1]))

        for i in range(1, mask.shape[2] + 1):

            tmp = np.where(mask[:,:,i-1] > 0, i, 0)

            out_mask += tmp

        out_mask = torch.tensor(out_mask)

        #mask = torch.tensor(mask)

        #mask = mask.float()

        #mask = mask/255

        

        return img, out_mask



    def __len__(self):

        return int(self.total_ims)
dataset = LunarDataset(render_list, clean_list)
testimg, testmask = dataset.__getitem__(10)

testimg.shape, testmask.shape
#_, single_mask = torch.max(testmask.view(480, 720,3), dim=2)

#single_mask
#tmp = torch.where(testmask.view(480, 720, 3)[:,:,2] > 0, torch.tensor(1), torch.tensor(0))

#single_mask += tmp
plt.imshow(testimg.view(480, 720,3))
plt.imshow(testmask)
# Check class indexes

#single_mask = torch.where(single_mask == 2, torch.tensor(1), torch.tensor(0))
#plt.imshow(single_mask)
testmask.unique()
test_im = testimg.unsqueeze(0)

test_mask = testmask.unsqueeze(0)
model_ft = models.resnet18(pretrained=False)
list(model_ft.children())
class Encoder(nn.Module):

    def __init__(self, model):

        super(Encoder, self).__init__()

        #self.init_model = model

        self.tmp_model = list(model.children())[:-2]

        self.block1 = nn.Sequential(*self.tmp_model[:3])

        self.block2 = nn.Sequential(*self.tmp_model[3:5])

        self.block3 = nn.Sequential(*self.tmp_model[5])

        self.block4 = nn.Sequential(*self.tmp_model[6])

        self.block5 = nn.Sequential(*self.tmp_model[7])

           

    def forward(self, x):

        x = self.block1(x)

        x1 = x

        x = self.block2(x)

        x2 = x

        x = self.block3(x)

        x3 = x

        x = self.block4(x)

        x4 = x

        x = self.block5(x)

        x5 = x

        

        intermediate_outs = [x1,x2,x3,x4,x5]

        return x, intermediate_outs
sample_encoder = Encoder(model_ft)

sample_encoder = sample_encoder.to(device)
sample_enc_out, layer_outputs = sample_encoder(test_im.to(device))

sample_enc_out.shape, layer_outputs[3].shape
class Decoder(nn.Module):

    def __init__(self, out_classes):

        self.out_classes = out_classes

        super(Decoder, self).__init__()

        #self.upsample0 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size = 2, stride=2),nn.BatchNorm2d(256), nn.ReLU(inplace=True)) #(512, 8, 100)

        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size = (2,3), stride=2, padding=(0,1)),nn.BatchNorm2d(256), nn.ReLU(inplace=True)) #(256 , 16, 100)

        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(256 + 256, 128, kernel_size = 2, stride=2),nn.BatchNorm2d(128), nn.ReLU(inplace=True)) #(128, 32, 200)

        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, kernel_size = 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) #(64, 64, 400)

        self.upsample4 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, kernel_size = 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) #(64, 128, 800)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.dilated1 = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size = 3, stride = 1, dilation=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.out = nn.Sequential(nn.ConvTranspose2d(64, self.out_classes, kernel_size = 2, stride=2), nn.LogSoftmax(dim=1)) #(classes, 256, 1600)

        self.skip1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size = 1, stride = 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.skip2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 1, stride = 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.skip3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 1, stride = 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.skip4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 1, stride = 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        

    def forward(self, x, skips):

        #x = torch.cat([x, skips[4]], dim=1)

        #x = self.upsample0(x)

        #x += skips[3]

        #x = torch.cat([x, skips[3]], dim=1)

        x = self.upsample1(x)

        skip1 = self.skip1(skips[3])

        x = torch.cat([x, skip1], dim=1)

        #x = self.conv1(x)

        x = self.upsample2(x)

        skip2 = self.skip2(skips[2])

        x = torch.cat([x, skip2], dim=1)

        #x = self.conv2(x)

        x = self.upsample3(x)

        skip3 = self.skip3(skips[1])

        x = torch.cat([x, skip3], dim=1)

        #x = self.conv3(x)

        x = self.upsample4(x)

        skip4 = self.skip4(skips[0])

        x = torch.cat([x, skip4], dim=1)

        x = self.conv4(x)

        #x = self.dilated1(x)

        

        out = self.out(x)

        return out
sample_decoder= Decoder(4)

sample_decoder = sample_decoder.to(device)

sample_decoder_output = sample_decoder(sample_enc_out, layer_outputs)
sample_decoder_output.size()
class Unet(nn.Module):

    def __init__(self, out_classes, model):

        super(Unet, self).__init__()

        self.out_classes = out_classes

        #self.model = model

        self.encoder = Encoder(model)

        self.decoder = Decoder(self.out_classes)

        

    def forward(self, x):

        enc_output, layer_values = self.encoder(x)

        

        dec_output = self.decoder(enc_output, layer_values)

        

        return dec_output
sample_unet = Unet(4, model_ft)

sample_unet = sample_unet.to(device)
final_unet = sample_unet(test_im.to(device))

final_unet.shape
final_unet = torch.argmax(final_unet, dim=1, keepdim=True)

final_unet.shape
final_unet = final_unet.squeeze(1)

final_unet.shape
plt.imshow(final_unet.view(480,720,1).cpu()[:,:,0])
indices = torch.randperm(len(dataset)).tolist()

train_size = int(len(indices)*0.8)

test_size = int(len(indices)*0.2)

train_size, test_size
train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])

test_dataset = torch.utils.data.Subset(dataset, indices[-test_size:])
#test_dataset.dataset.transform = transforms.Compose([

#    transforms.Normalize(mean=[0.485, 0.456, 0.406],

#                         std=[0.229, 0.224, 0.225])

#])



train_dataset.dataset.transform = transforms.Compose([

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                         std=[0.229, 0.224, 0.225]),

])
train_loader = torch.utils.data.DataLoader(

        train_dataset, batch_size=16, shuffle=True)



test_loader = torch.utils.data.DataLoader(

        test_dataset, batch_size=32, shuffle=False)
def dice_loss(inputs, labels):

    inputs = torch.argmax(inputs, dim=1, keepdim = True)

    inputs = inputs.squeeze(1)

    labels = labels.squeeze(1)

    intersection = (inputs.view(-1).float() == labels.view(-1).float())

    sum_int = intersection.sum().float()



    inp_flats = inputs.view(-1).float()

    lab_flats = labels.view(-1).float()    



    d_loss = (2 * sum_int/(len(inp_flats) + len(lab_flats)))

    return d_loss
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=5):

    since = time.time()



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        model.train()



        running_loss = 0.0

        running_acc = 0.0

        batch_loss = 0.0

        batch_acc = 0.0

        batch = 0

        #batch_no = 1



            # Iterate over data.

        for inputs, labels in train_loader:

            batch = batch + 1

            inputs = inputs.to(device)

            labels = labels.float()

            labels = labels.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward

            # track history if only in train

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                loss = criterion(outputs.float(), labels.long())



                # backward + optimize only if in training phase

                loss.backward()

                optimizer.step()



            # statistics

            running_loss += loss.item() * inputs.size(0)

            acc = dice_loss(outputs, labels)

            running_acc += acc.item() * inputs.size(0)

            batch_loss += loss.item() * inputs.size(0)

            batch_acc += acc.item()

            

            if batch % 25 == 0:

                batch_loss = batch_loss/(16*25)

                batch_acc = batch_acc/(25)

                print('Batch {}, Batch Loss: {:.4f}, Dice Coeff {:.4}'.format(batch, batch_loss, batch_acc))

                batch_acc = 0

                batch_loss = 0

                

        epoch_acc = running_acc/len(train_dataset)

        epoch_loss = running_loss / len(train_dataset)

        

        model.eval()

        vrunning_loss = 0.0

        vrunning_acc = 0.0

        for vinputs, vlabels in test_loader:

            vinputs = vinputs.to(device)

            vlabels = vlabels.float().to(device)

            with torch.no_grad():

                voutputs = model(vinputs)

                vloss = criterion(voutputs.float(), vlabels.long())

                vrunning_loss += vloss.item() * vinputs.size(0)

                v_acc = dice_loss(voutputs, vlabels)

                vrunning_acc += v_acc.item() * vinputs.size(0)

            

        vepoch_acc = vrunning_acc/len(test_dataset)

        vepoch_loss = vrunning_loss / len(test_dataset)

            

        print('{} Epoch Loss: {:.4f}'.format(epoch, epoch_loss))

        print('Validation: {} Epoch Val Loss: {:.4f} Epoch Val acc {:.4}'.format(epoch, vepoch_loss, vepoch_acc))

        

        test_real, test_mask = test_dataset.__getitem__(0)

        test_real = test_real.unsqueeze(0).to(device)

        test_mask = test_mask.unsqueeze(0).to(device)

        test_out = model(test_real)

        final_out = torch.argmax(test_out, dim=1, keepdim=True)

        final_out = final_out.squeeze(0)

        test_real = test_real.squeeze(0)

        test_mask = test_mask.squeeze(0)

        fig, axes = plt.subplots(1,3, figsize=(10,10))

        axes[0].set_title('Real Image')

        axes[0].imshow(test_real.view((480, 720,3)).cpu())

        axes[1].set_title('Ground Truth')

        axes[1].imshow(test_mask.view((480, 720)).cpu())

        axes[2].set_title('Model Prediction')

        axes[2].imshow(final_out.view(480,720,1).cpu()[:,:,0])



        print()

        scheduler.step()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    return model
Unet_model = Unet(4, model_ft)
for params in Unet_model.parameters():

    params.requires_grad = True
Unet_model = Unet_model.to(device)



criterion = nn.NLLLoss()

#criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, Unet_model.parameters()), lr=0.009, momentum=0.9)



# Decay LR by a factor of 0.1 every 1 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.001)
model_res = train_model(Unet_model, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)