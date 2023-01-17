# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread as imread

from skimage.transform import resize as resize

from glob import glob

from tqdm import tqdm

import cv2

import gc



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Variable



from albumentations import Compose, ShiftScaleRotate, Rotate, RandomScale



import matplotlib.pyplot as plt

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
image_paths = glob(os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities', 'Montgomery', 'MontgomerySet', 'CXR_png', '*.png'))
images_with_masks_paths = [(image_path,

                      os.path.join('/'.join(image_path.split('/')[:-2]), 'ManualMask', 'leftMask', os.path.basename(image_path)), 

                      os.path.join('/'.join(image_path.split('/')[:-2]), 'ManualMask', 'rightMask', os.path.basename(image_path))) for image_path in image_paths]
images_with_masks_paths[0]
cv2.imread(images_with_masks_paths[40][0]).shape
OUT_DIM = (512, 512)
def image_from_path(path):

    img = resize(imread(path), OUT_DIM, mode='constant')

    return img



def mask_from_paths(path1, path2):

    img = resize(cv2.bitwise_or(imread(path1), imread(path2)), OUT_DIM, mode='constant')

    return img
images = []

masks = []



for mri, left_lung, right_lung in tqdm(images_with_masks_paths, position=0, leave=True):

    images.append(image_from_path(mri))

    masks.append(mask_from_paths(left_lung, right_lung))
def random_plot(images, masks, number):

    indices = np.random.choice(len(images), number)

    fig, axis = plt.subplots(nrows=number, ncols=3, figsize=(20, 20))



    for i, index in enumerate(indices):

        img = images[index]

        mask = masks[index]



        axis[i][0].imshow(img, cmap='gray')

        axis[i][1].imshow(mask, cmap='gray')

        axis[i][2].imshow(cv2.addWeighted(img, 1.0, mask, 0.7, 1), cmap='gray')



    plt.tight_layout()

    

# random_plot(images, masks, 4)
transform = Compose([

    ShiftScaleRotate(rotate_limit=15, always_apply=True)

])
transformed_images = []

transformed_masks = []



for image, mask in zip(images, masks):

    sample = {'image': image.copy(), 'mask': mask.copy()}

    out = transform(**sample)

    transformed_images.append(out['image'])

    transformed_masks.append(out['mask'])
# random_plot(transformed_images, transformed_masks, 4)
from sklearn.model_selection import train_test_split



image_dataset = images.copy() + transformed_images

mask_dataset = masks.copy() + transformed_masks



X_train, X_val, y_train, y_val = train_test_split(image_dataset, mask_dataset, test_size=0.2)
from sklearn.preprocessing import StandardScaler



scaler_X = StandardScaler()

scaler_y = StandardScaler()



X_train = scaler_X.fit_transform(np.array(X_train).reshape(-1, 512 * 512)).reshape(-1, 512, 512)

# y_train = scaler_y.fit_transform(np.array(y_train).reshape(-1, 512 * 512)).reshape(-1, 512, 512)



X_val = scaler_X.transform(np.array(X_val).reshape(-1, 512 * 512)).reshape(-1, 512, 512)

# y_val = scaler_y.transform(np.array(y_val).reshape(-1, 512 * 512)).reshape(-1, 512, 512)
batch_size = 4



train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))



train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
# ref: https://github.com/jvanvugt/pytorch-unet

#      https://github.com/jaxony/unet-pytorch



def conv1x1(in_channels, out_channels, groups=1):

    return nn.Conv2d(in_channels,

                     out_channels,

                     kernel_size=1,

                     groups=groups,

                     stride=1)



def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):

    return nn.Conv2d(in_channels,

                     out_channels,

                     kernel_size=3,

                     stride=stride,

                     padding=padding,

                     bias=bias,

                     groups=groups)



def upconv2x2(in_channels, out_channels, mode='transpose'):

    if mode == 'transpose':

        return nn.ConvTranspose2d(in_channels,

                                  out_channels,

                                  kernel_size=2,

                                  stride=2)

    else:

        return nn.Sequential(

            nn.Upsample(mode='bilinear', scale_factor=2),

            conv1x1(in_channels, out_channels))
class DownConv(nn.Module):

    """

    A helper Module that performs 2 convolutions and 1 MaxPool.

    A ReLU activation follows each convolution.

    """

    def __init__(self, in_channels, out_channels, pooling=True):

        super(DownConv, self).__init__()



        self.in_channels = in_channels

        self.out_channels = out_channels

        self.pooling = pooling



        self.conv1 = conv3x3(self.in_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)



        if self.pooling:

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        before_pool = x

        if self.pooling:

            x = self.pool(x)

        return x, before_pool



class UpConv(nn.Module):

    """

    A helper Module that performs 2 convolutions and 1 UpConvolution.

    A ReLU activation follows each convolution.

    """

    def __init__(self,

                 in_channels,

                 out_channels,

                 merge_mode='concat',

                 up_mode='transpose'):

        super(UpConv, self).__init__()



        self.in_channels = in_channels

        self.out_channels = out_channels

        self.merge_mode = merge_mode

        self.up_mode = up_mode



        self.upconv = upconv2x2(self.in_channels,

                                self.out_channels,

                                mode=self.up_mode)



        if self.merge_mode == 'concat':

            self.conv1 = conv3x3(2*self.out_channels,

                                 self.out_channels)

        else:

            # num of input channels to conv2 is same

            self.conv1 = conv3x3(self.out_channels, self.out_channels)



        self.conv2 = conv3x3(self.out_channels, self.out_channels)



    def forward(self, from_down, from_up):

        """ Forward pass

        Arguments:

            from_down: tensor from the encoder pathway

            from_up: upconv'd tensor from the decoder pathway

        """

        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':

            x = torch.cat((from_up, from_down), 1)

        else:

            x = from_up + from_down

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        return x
class UNet(nn.Module):

    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.

    Contextual spatial information (from the decoding,

    expansive pathway) about an input tensor is merged with

    information representing the localization of details

    (from the encoding, compressive pathway).

    Modifications to the original paper:

    (1) padding is used in 3x3 convolutions to prevent loss

        of border pixels

    (2) merging outputs does not require cropping due to (1)

    (3) residual connections can be used by specifying

        UNet(merge_mode='add')

    (4) if non-parametric upsampling is used in the decoder

        pathway (specified by upmode='upsample'), then an

        additional 1x1 2d convolution occurs after upsampling

        to reduce channel dimensionality by a factor of 2.

        This channel halving happens with the convolution in

        the tranpose convolution (specified by upmode='transpose')

    """



    def __init__(self, num_classes, in_channels=3, depth=5,

                 start_filts=64, up_mode='transpose',

                 merge_mode='concat'):

        """

        Arguments:

            in_channels: int, number of channels in the input tensor.

                Default is 3 for RGB images.

            depth: int, number of MaxPools in the U-Net.

            start_filts: int, number of convolutional filters for the

                first conv.

            up_mode: string, type of upconvolution. Choices: 'transpose'

                for transpose convolution or 'upsample' for nearest neighbour

                upsampling.

        """

        super(UNet, self).__init__()



        if up_mode in ('transpose', 'upsample'):

            self.up_mode = up_mode

        else:

            raise ValueError("\"{}\" is not a valid mode for "

                             "upsampling. Only \"transpose\" and "

                             "\"upsample\" are allowed.".format(up_mode))



        if merge_mode in ('concat', 'add'):

            self.merge_mode = merge_mode

        else:

            raise ValueError("\"{}\" is not a valid mode for"

                             "merging up and down paths. "

                             "Only \"concat\" and "

                             "\"add\" are allowed.".format(up_mode))



        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'

        if self.up_mode == 'upsample' and self.merge_mode == 'add':

            raise ValueError("up_mode \"upsample\" is incompatible "

                             "with merge_mode \"add\" at the moment "

                             "because it doesn't make sense to use "

                             "nearest neighbour to reduce "

                             "depth channels (by half).")



        self.num_classes = num_classes

        self.in_channels = in_channels

        self.start_filts = start_filts

        self.depth = depth



        self.down_convs = []

        self.up_convs = []



        # create the encoder pathway and add to a list

        for i in range(depth):

            ins = self.in_channels if i == 0 else outs

            outs = self.start_filts*(2**i)

            pooling = True if i < depth-1 else False



            down_conv = DownConv(ins, outs, pooling=pooling)

            self.down_convs.append(down_conv)



        # create the decoder pathway and add to a list

        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):

            ins = outs

            outs = ins // 2

            up_conv = UpConv(ins, outs, up_mode=up_mode,

                merge_mode=merge_mode)

            self.up_convs.append(up_conv)



        self.conv_final = conv1x1(outs, self.num_classes)



        # add the list of modules to current module

        self.down_convs = nn.ModuleList(self.down_convs)

        self.up_convs = nn.ModuleList(self.up_convs)



        self.reset_params()



    @staticmethod

    def weight_init(m):

        if isinstance(m, nn.Conv2d):

            nn.init.xavier_normal_(m.weight)

            nn.init.constant_(m.bias, 0)





    def reset_params(self):

        for i, m in enumerate(self.modules()):

            self.weight_init(m)



    def forward(self, x):

        encoder_outs = []



        # encoder pathway, save outputs for merging

        for i, module in enumerate(self.down_convs):

            x, before_pool = module(x)

            encoder_outs.append(before_pool)



        for i, module in enumerate(self.up_convs):

            before_pool = encoder_outs[-(i+2)]

            x = module(before_pool, x)



        # No softmax is used. This means you need to use

        # nn.CrossEntropyLoss is your training script,

        # as this module includes a softmax already.

        x = self.conv_final(x)

        return x
lr = 0.0001



model = UNet(2, depth=5, start_filts=64, in_channels=1).cuda()

critertion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch.cuda.empty_cache()

gc.collect()
loss_history = []

epochs = 50



for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):

        inputs = Variable(images.view(batch_size, 1, 512, 512)).cuda()

        labels = Variable(np.round(labels).view(batch_size, 512, 512)).cuda()



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = model(inputs.float())

        loss = critertion(outputs, labels.long())

        loss.backward()

        optimizer.step()

        

        # print statistics

        running_loss += loss.item()

        loss_history.append(loss.item())

        if i % 10 == 9:    # print every 10 mini-batches

            print('[%d, %5d] loss: %.10f' %

                  (epoch + 1, i + 1, running_loss / 10))

            running_loss = 0.0



print('Finished Training')
fig = plt.figure(figsize=(10, 6))

plt.plot(list(range(len(loss_history))), loss_history)
torch.save(model, './model_e50_lr0.0002.ser')
pred = []

with torch.no_grad():

    for sample, label in zip(X_val, y_val):

        pred += model(Variable(torch.Tensor(sample)).view(1, 1, 512, 512).cuda())
indices = np.random.choice(len(X_val), 5)

fig, axis = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))



for i, index in enumerate(indices):

    img = X_val[index]

    mask = pred[index][0, :, :]

    true_mask = y_val[index]

    mask = cv2.bitwise_not(cv2.cvtColor(mask.cpu().detach().numpy(), cv2.COLOR_BGR2RGB))

    mask[mask > 0] = 255

    

    axis[i][0].imshow(img, cmap='gray')

    axis[i][1].imshow(mask)

    axis[i][2].imshow(true_mask, cmap='gray')



plt.tight_layout()

# # ref: https://github.com/pytorch/pytorch/issues/1249

# def dice_loss(pred, target):

#     smooth = 1.



#     iflat = torch.Tensor(np.array(pred)).view(-1)

#     tflat = target.view(-1)

#     intersection = np.sum(iflat * tflat)



#     return 1 - ((2. * intersection + smooth) / (np.sum(iflat) + np.sum(tflat) + smooth))





# print(dice_loss(pred, torch.Tensor(y_val)))