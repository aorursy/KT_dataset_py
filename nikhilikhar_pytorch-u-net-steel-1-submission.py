import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook as tqdm



import torch

import torch.nn as nn

from torch import optim

import torchvision.transforms as transforms

from torchvision import models

import torch.nn.functional as F

from torch.autograd import Function, Variable

from pathlib import Path

from itertools import groupby

import time


start = time.time()



input_dir = "../input/severstal-steel-defect-detection/"

train_img_dir = "../input/severstal-steel-defect-detection/train_images/"

test_img_dir = "../input/severstal-steel-defect-detection/test_images/"



category_num = 4 + 1



ratio = 1

epoch_num = 1

batch_size = 2

device = "cuda:0"
train_df = pd.read_csv(input_dir + "train.csv")

train_df[['ImageId', 'ClassId']] = train_df['ImageId_ClassId'].str.split('_', expand=True)

train_df.head()
train_df.shape
def make_mask_img(segment_df):

    seg_width = 1600

    seg_height = 256

    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        if pd.isna(encoded_pixels): continue

        pixel_list = list(map(int, encoded_pixels.split(" ")))

        for i in range(0, len(pixel_list), 2):

            start_index = pixel_list[i] -1 

            index_len = pixel_list[i+1] 

            seg_img[start_index:start_index+index_len] = int(class_id) 

    seg_img = seg_img.reshape((seg_height, seg_width), order='F')

   

    return seg_img
def train_generator(df, batch_size):

    img_ind_num = df.groupby("ImageId")["ClassId"].count()

    index = df.index.values[0]

    trn_images = []

    seg_images = []

    for i, (img_name, ind_num) in enumerate(img_ind_num.items()):

        img = cv2.imread(train_img_dir + img_name)

        segment_df = (df.loc[index:index+ind_num-1, :]).reset_index(drop=True)

        index += ind_num

        if segment_df["ImageId"].nunique() != 1:

            raise Exception("Index Range Error")

        seg_img = make_mask_img(segment_df)

        

        # HWC -> CHW

        img = img.transpose((2, 0, 1))

        #seg_img = seg_img.transpose((2, 0, 1))

        

        trn_images.append(img)

        seg_images.append(seg_img)

        if((i+1) % batch_size == 0):

            yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)

            trn_images = []

            seg_images = []

    if(len(trn_images) != 0):

        yield np.array(trn_images, dtype=np.float32) / 255, np.array(seg_images, dtype=np.int32)
def test_generator(img_names):

    for img_name in img_names:

        img = cv2.imread(test_img_dir + img_name)

        # HWC -> CHW

        img = img.transpose((2, 0, 1))

        yield img_name, np.asarray([img], dtype=np.float32) / 255
def encode(input_string):

    return [(len(list(g)), k) for k,g in groupby(input_string)]



def run_length(label_vec):

    encode_list = encode(label_vec)

    index = 1

    class_dict = {}

    for i in encode_list:

        if i[1] != category_num-1:

            if i[1] not in class_dict.keys():

                class_dict[i[1]] = []

            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]

        index += i[0]

    return class_dict
class double_conv(nn.Module):

    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        x = self.conv(x)

        return x





class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x):

        x = self.conv(x)

        return x





class down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(down, self).__init__()

        self.mpconv = nn.Sequential(

            nn.MaxPool2d(2),

            double_conv(in_ch, out_ch)

        )



    def forward(self, x):

        x = self.mpconv(x)

        return x





class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super(up, self).__init__()



        #  would be a nice idea if the upsampling could be learned too,

        #  but my machine do not have enough memory to handle all those weights

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)



        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffX = x1.size()[2] - x2.size()[2]

        diffY = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),

                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x





class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)



    def forward(self, x):

        x = self.conv(x)

        return x



    

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):

        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)

        self.down1 = down(64, 128)

        self.down2 = down(128, 256)

        self.down3 = down(256, 512)

        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)

        self.up2 = up(512, 128)

        self.up3 = up(256, 64)

        self.up4 = up(128, 64)

        self.outc = outconv(64, n_classes)



    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outc(x)

        return x
# https://github.com/usuyama/pytorch-unet



def convrelu(in_channels, out_channels, kernel, padding):

    return nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),

        nn.ReLU(inplace=True),

    )





class ResNetUNet(nn.Module):

    def __init__(self, n_class):

        super().__init__()



        self.base_model = models.resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())



        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)

        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)

        self.layer1_1x1 = convrelu(64, 64, 1, 0)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)

        self.layer2_1x1 = convrelu(128, 128, 1, 0)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)

        self.layer3_1x1 = convrelu(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.layer4_1x1 = convrelu(512, 512, 1, 0)



        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)



        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)

        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)

        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)

        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)



        self.conv_original_size0 = convrelu(3, 64, 3, 1)

        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)



        self.conv_last = nn.Conv2d(64, n_class, 1)



    def forward(self, input):

        x_original = self.conv_original_size0(input)

        x_original = self.conv_original_size1(x_original)



        layer0 = self.layer0(input)

        layer1 = self.layer1(layer0)

        layer2 = self.layer2(layer1)

        layer3 = self.layer3(layer2)

        layer4 = self.layer4(layer3)



        layer4 = self.layer4_1x1(layer4)

        x = self.upsample(layer4)

        layer3 = self.layer3_1x1(layer3)

        x = torch.cat([x, layer3], dim=1)

        x = self.conv_up3(x)



        x = self.upsample(x)

        layer2 = self.layer2_1x1(layer2)

        x = torch.cat([x, layer2], dim=1)

        x = self.conv_up2(x)



        x = self.upsample(x)

        layer1 = self.layer1_1x1(layer1)

        x = torch.cat([x, layer1], dim=1)

        x = self.conv_up1(x)



        x = self.upsample(x)

        layer0 = self.layer0_1x1(layer0)

        x = torch.cat([x, layer0], dim=1)

        x = self.conv_up0(x)



        x = self.upsample(x)

        x = torch.cat([x, x_original], dim=1)

        x = self.conv_original_size2(x)



        out = self.conv_last(x)



        return out
!mkdir -p /tmp/.cache/torch/checkpoints/

!cp ../input/resnet18/resnet18.pth /tmp/.cache/torch/checkpoints/resnet18-5c106cde.pth
# net = UNet(n_channels=3, n_classes=category_num).to(device)

net = ResNetUNet(n_class=category_num).to(device)



optimizer = optim.SGD(

    net.parameters(),

    lr=0.1,

    momentum=0.9,

    weight_decay=0.0005

)



criterion = nn.CrossEntropyLoss()

train_df.shape


checkpoint = torch.load(Path('../input/u-net-baseline-by-pytorch-steel/model-exported'))

net.load_state_dict(checkpoint)
print("Total length of train df {}".format(len(train_df)))
# val_sta = 40000

# val_end = 50000

# train_loss = []

# valid_loss = []

# for epoch in range(epoch_num):

#     epoch_trn_loss = 0

#     train_len = 0

#     net.train()

#     for iteration, (X_trn, Y_trn) in enumerate(tqdm(train_generator(train_df.iloc[:val_sta, :], batch_size))):

#         X = torch.tensor(X_trn, dtype=torch.float32).to(device)

#         Y = torch.tensor(Y_trn, dtype=torch.long).to(device)

#         train_len += len(X)

        

#         #Y_flat = Y.view(-1)

#         mask_pred = net(X)

#         #mask_prob = torch.softmax(mask_pred, dim=1)

#         #mask_prob_flat = mask_prob.view(-1)

#         loss = criterion(mask_pred, Y)

#         optimizer.zero_grad()

#         loss.backward()

#         optimizer.step()

#         epoch_trn_loss += loss.item()

        

#         if iteration % 100 == 0:

#             print("train loss in {:0>2}epoch  /{:>5}iter:    {:<10.8}".format(epoch+1, iteration, epoch_trn_loss/(iteration+1)))

        

#     train_loss.append(epoch_trn_loss/(iteration+1))

#     print("train {}epoch loss({}iteration):    {:10.8}".format(epoch+1, iteration, train_loss[-1]))

    

#     epoch_val_loss = 0

#     val_len = 0

#     net.eval()

#     for iteration, (X_val, Y_val) in enumerate(tqdm(train_generator(train_df.iloc[val_sta:val_end, :], batch_size))):

#         X = torch.tensor(X_val, dtype=torch.float32).to(device)

#         Y = torch.tensor(Y_val, dtype=torch.long).to(device)

#         val_len += len(X)

        

#         #Y_flat = Y.view(-1)

        

#         mask_pred = net(X)

#         #mask_prob = torch.softmax(mask_pred, dim=1)

#         #mask_prob_flat = mask_prob.view(-1)

#         loss = criterion(mask_pred, Y)

#         epoch_val_loss += loss.item()

        

#         if iteration % 100 == 0:

#             print("valid loss in {:0>2}epoch  /{:>5}iter:    {:<10.8}".format(epoch+1, iteration, epoch_val_loss/(iteration+1)))

        

#     valid_loss.append(epoch_val_loss/(iteration+1))

#     print("valid {}epoch loss({}iteration):    {:10.8}".format(epoch+1, iteration, valid_loss[-1]))
# plt.plot(list(range(epoch_num)), train_loss, color='green')

# plt.plot(list(range(epoch_num)), valid_loss, color='blue')
# torch.save(net.state_dict(), './model-exported')
sample_df = pd.read_csv(input_dir + "sample_submission.csv")

sample_df[['ImageId', 'ClassId']] = sample_df['ImageId_ClassId'].str.split('_', expand=True)

sample_df.head()
# import torch

# import gc

# for obj in gc.get_objects():

#     try:

#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):

#             print(type(obj), obj.size())

#     except:

#         pass
len(sample_df)
sub_list = []

net.eval()

test_images = sample_df["ImageId"].unique()

for img_name, img in tqdm(test_generator(test_images), total=len(test_images)):

    X = torch.tensor(img, dtype=torch.float32).to(device)

    mask_pred = net(X)

    mask_pred = mask_pred.cpu().detach().numpy()

    mask_prob = np.argmax(mask_pred, axis=1)

    mask_prob = mask_prob.T.ravel(order='F')

    class_dict = run_length(mask_prob)

    if len(class_dict) == 0:

        for i in range(4):

            sub_list.append([img_name+ "_" + str(i+1), ''])

    else:

        for key, val in class_dict.items():

            sub_list.append([img_name + "_" + str(key+1), " ".join(map(str, val))])

        for i in range(4):

            if i not in class_dict.keys():

                sub_list.append([img_name+ "_" + str(i+1), ''])

                

print("Total len {0}".format(len(sub_list)))

print(sub_list[:5])

# img_name = '5e581254c.jpg'

# img = cv2.imread(train_img_dir + img_name)

# # HWC -> CHW

# img = img.transpose((2, 0, 1))

# img = np.asarray([img], dtype=np.float32) / 255

# X = torch.tensor(img, dtype=torch.float32).to(device)

# mask_pred = net(X)

# mask_pred = mask_pred.cpu().detach().numpy()

# mask_prob = np.argmax(mask_pred, axis=1)

# mask_prob = mask_prob.ravel()

# mask_prob

# mask_prob.resize(256, 1600)

# plt.imshow(mask_prob)



# d = run_length(mask_prob.ravel())

# nmask = {}

# nmask['EncodedPixels'] = []

# nmask['ClassId'] = []

# for k,v in d.items():

#     nmask['ClassId'].append(str(k))

#     nmask['EncodedPixels'].append(' '.join(map(str,v)))

# for i in range(4):

#     if str(i) not in nmask['ClassId']:

#         nmask['ClassId'].append(str(i))

#         nmask['EncodedPixels'].append(np.nan)

# nmask = pd.DataFrame.from_dict(nmask)

# nmask
submission_df = pd.DataFrame(sub_list, columns=['ImageId_ClassId', 'EncodedPixels'])
submission_df.head()
submission_df.to_csv("submission.csv", index=False)
end = time.time()

hours, rem = divmod(end-start, 3600)

minutes, seconds = divmod(rem, 60)

print("Execution Time  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))