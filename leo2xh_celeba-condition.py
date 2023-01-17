import os

import random

import time



import cv2

import matplotlib

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import torch

import skimage

from tqdm import tqdm

from skimage import io, transform

from torch.utils import data

from torchvision import transforms

from matplotlib import pyplot as plt

import torch.nn as nn

import torch

import torch

import torch.nn as nn

from torch.nn import init

import functools

from torch.optim import lr_scheduler



class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,

                 padding_type='reflect', direction = 'H2L', n_attribute = 6):

        assert(n_blocks >= 0)

        super(ResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:

            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:

            use_bias = norm_layer == nn.InstanceNorm2d



        assert((direction == 'H2L') or (direction == 'L2H'))

        

        n_downsampling = 2

        

        if direction != 'H2L':

            input_nc += n_attribute

        

        model = [nn.ReflectionPad2d(3),

                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),

                 norm_layer(ngf),

                 nn.ReLU(True)]



        

        for i in range(n_downsampling):  # add downsampling layers

            mult = 2 ** i

            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),

                      norm_layer(ngf * mult * 2),

                      nn.ReLU(True)]



        mult = 2 ** n_downsampling

        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]



        

        if direction != 'H2L':

            for i in range(n_downsampling * 2):  # add upsampling layers

                mult = 2 ** (n_downsampling - i)               

                model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),

                                             kernel_size=3, stride=2,

                                             padding=1, output_padding=1,

                                             bias=use_bias),

                          norm_layer(int(ngf * mult / 2)),

                          nn.ReLU(True)]

            ngf = int(ngf * mult / 2)

        else:

            ngf *= (1 << n_downsampling)

    

        model += [nn.ReflectionPad2d(3)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        model += [nn.Tanh()]

        

        self.model = nn.Sequential(*model)

        

    def label2matrix(self, bs, h, w, z):

        _, c = z.shape

        res = torch.zeros((bs, c, h, w))

        for index, c_ in enumerate(z):

            for index_, value in enumerate(c_):

                if value == 1:

                    res[index][index_] = torch.ones((h, w))

        return res



    

    def forward(self, input, label = None, device = None):

        """Standard forward"""

        bs, _, h, w = input.shape

        if type(label) == torch.Tensor:

            tmp = self.label2matrix(bs, h, w, label).cuda(device)

            input = torch.cat((input, tmp), 1)

        return self.model(input)



class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        """Define a Resnet block"""

        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)



    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []

        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)



        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        if use_dropout:

            conv_block += [nn.Dropout(0.5)]



        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        

        return nn.Sequential(*conv_block)



    def forward(self, x):

        """Standard forward"""

        out = x + self.conv_block(x)  # add skip connections

        return out

    



class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, img_size, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 

                 add_channel = False, n_attributes = None):

        

        """Create a Discriminator 

            

            Parameters:

                input_nc (int) -- the number of channels in input images

                img_size (int) -- the size of input images

                ngf (int) -- the number of filters in the last conv layer

                n_layers (int)  -- the number of conv layers in the discriminator

                norm (str) -- the name of normalization layers used in the network: batch | instance | none

                add_channel -- if use label to help discriminate: True | false

                n_attribute  -- the number of attributes in input images, only necessary when add_channel is True

                

            Our current implementation is based on the work from junyanz. See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix for more details

            We add extra parameters and fc layers to help calculate the probability for arbitrary input with different size.

            """

        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters

            use_bias = norm_layer.func != nn.BatchNorm2d

        else:

            use_bias = norm_layer != nn.BatchNorm2d



        kw = 4

        padw = 1

        

        sequence_1 = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        self.model_1 = nn.Sequential(*sequence_1)

        

        nf_mult = 1

        nf_mult_prev = 1

        if add_channel == True:

            ndf += n_attributes

            

        sequence_2 = []

        for n in range(1, n_layers):  # gradually increase the number of filters

            nf_mult_prev = nf_mult

            nf_mult = min(2 ** n, 8)

            sequence_2 += [

                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),

                norm_layer(ndf * nf_mult),

                nn.LeakyReLU(0.2, True)

            ]



        nf_mult_prev = nf_mult

        nf_mult = min(2 ** n_layers, 8)

        sequence_2 += [

            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),

            norm_layer(ndf * nf_mult),

            nn.LeakyReLU(0.2, True)

        ]



        sequence_2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        self.model_2 = nn.Sequential(*sequence_2)

    

        sequence_3 = []

        sequence_3 += [nn.Linear(self.get_size(img_size, n_layers), ndf), nn.Linear(ndf, 1)] 

        self.model_3 = nn.Sequential(*sequence_3)

        

    def get_size(self, x, n_layers):

        """specificify the cahnenl for speicific input"""

        h, w = x

        for i in range(n_layers):

            h = int(h/2)

            w = int(w/2)

        for i in range(2):

            w -= 1

            h -= 1

        return h*w

        

    def label2matrix(self, bs, h, w, z):

        _, c = z.shape

        res = torch.zeros((bs, c, h, w))

        for index, c_ in enumerate(z):

            for index_, value in enumerate(c_):

                if value == 1:

                    res[index][index_] = torch.ones((h, w))

        return res

        

    def num_flat_features(self, x):

        """Return number of total elements in feature maps"""

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features

    

    def forward(self, input, label = None, device = None):

        """Standard forward."""

        output = self.model_1(input)

        bs, _, h, w = output.shape

        

        if type(label) == torch.Tensor:

            tmp = self.label2matrix(bs, h, w, label).cuda(device)

            output = torch.cat((output, tmp), 1)

        

        output = self.model_2(output)

        output = output.view(-1,self.num_flat_features(output))

        output = self.model_3(output)

        output = torch.nn.Sigmoid()(output)

        

        return output
def init_weights(net, init_type='normal', init_gain=0.02):

    """Initialize network weights.

    Parameters:

        net (network)   -- network to be initialized

        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal

        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might

    work better for some applications. Feel free to try yourself.

    """



    def init_func(m):  # define the initialization function

        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':

                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':

                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':

                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':

                init.orthogonal_(m.weight.data, gain=init_gain)

            else:

                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:

                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

            init.normal_(m.weight.data, 1.0, init_gain)

            init.constant_(m.bias.data, 0.0)



    print('initialize network with %s' % init_type)

    net.apply(init_func)  # apply the initialization function <init_func>
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights

    Parameters:

        net (network)      -- the network to be initialized

        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal

        gain (float)       -- scaling factor for normal, xavier and orthogonal.

        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.

    """

    if len(gpu_ids) > 0:

        assert(torch.cuda.is_available())

        net.to(gpu_ids[0])

        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)

    return net
class CeleaDataset(data.Dataset):

    def __init__(self, img_dir, label_path, attributes = [], test=False, scale = 1, spread=(None,None)):

        self.img_dir = img_dir

        self.images_list = sorted(os.path.splitext(img)[0] for img in list(os.walk(img_dir))[0][2])

        self.labels = pd.read_csv(label_path)[attributes] #筛选列

        self.labels = self.labels[self.sift(attributes)] #筛选行

        self.indexs = self.labels.index

        self.scale = scale

        if not spread == (None, None):

          self.indexs = self.indexs[spread[0]:spread[1]]

        

    def sift(self, attributes):

        assert len(attributes) > 0, "Please input the attributes"

        

        tmp = self.labels[attributes] != -1

        target = tmp[attributes[0]]

        for attr_ in attributes:

            target |= tmp[attr_]

        

        return target

    

    def __getitem__(self, index):

        img_index = self.indexs[index]

        img_path = os.path.join(self.img_dir, self.images_list[img_index]+".jpg")

        img = io.imread(img_path)

        size = 256 * self.scale

        img = skimage.transform.resize(img, (size, size))

        img = transforms.ToTensor()(img).float()

        

        label = torch.LongTensor(self.labels.loc[self.indexs[index]]).long()

        

        return img, label



    def __len__(self):

        return len(self.indexs)
X_label = ['Bald', 'Smiling', 'Blond_Hair', 'Eyeglasses', 'Pale_Skin' , 'Wearing_Hat']

Y_label = ['Bald', 'Smiling', 'Blond_Hair', 'Eyeglasses', 'Pale_Skin' , 'Wearing_Hat']
img_dir = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'

label_path = '../input/celeba-dataset/list_attr_celeba.csv'
!nvidia-smi
X_data = CeleaDataset(img_dir, label_path, X_label, spread=(0,20000))

Y_data = CeleaDataset(img_dir, label_path, Y_label, scale = 0.25, spread=(20000,40000))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
def init_func(m):  # define the initialization function

    classname = m.__class__.__name__

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        if init_type == 'normal':

            init.normal_(m.weight.data, 0.0, init_gain)

        elif init_type == 'xavier':

            init.xavier_normal_(m.weight.data, gain=init_gain)

        elif init_type == 'kaiming':

            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

        elif init_type == 'orthogonal':

            init.orthogonal_(m.weight.data, gain=init_gain)

        else:

            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

        if hasattr(m, 'bias') and m.bias is not None:

            init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

        init.normal_(m.weight.data, 1.0, init_gain)

        init.constant_(m.bias.data, 0.0)
G_X2Y = ResnetGenerator(3, 3, 64, nn.BatchNorm2d, True, 6, 'reflect', 'H2L', 6).cuda(device)

G_Y2X = ResnetGenerator(3, 3, 64, nn.BatchNorm2d, True, 6, 'reflect', 'L2H', 6).cuda(device)

D_X = NLayerDiscriminator(3, [256, 256], add_channel=True, n_attributes=6).cuda(device)

D_Y = NLayerDiscriminator(3, [64, 64]).cuda(device)
init_weights(G_X2Y,'kaiming' )

init_weights(G_Y2X, 'kaiming')

init_weights(D_X, 'kaiming')

init_weights(D_Y, 'kaiming')
optimizer_G_X2Y = torch.optim.Adam(G_X2Y.parameters(), lr=4e-6, betas=(0.9, 0.999))

optimizer_G_Y2X = torch.optim.Adam(G_Y2X.parameters(), lr=4e-6, betas=(0.9, 0.999))

optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=4e-6, betas=(0.9, 0.999))

optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=1e-7, betas=(0.9, 0.999))
def showRes(x,y, fake_x, fake_y, rec_x, rec_y):

    plt.figure()

    plt.subplot(2,3,1)

    x = transforms.ToPILImage()(x[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.subplot(2,3,2)

    x = transforms.ToPILImage()(fake_x[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.subplot(2,3,3)

    x = transforms.ToPILImage()(rec_x[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.subplot(2,3,4)

    x = transforms.ToPILImage()(y[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.subplot(2,3,5)

    x = transforms.ToPILImage()(fake_y[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.subplot(2,3,6)

    x = transforms.ToPILImage()(rec_y[0].cpu()).convert('RGB')

    plt.imshow(x)

    plt.show()
batch_size = 16

X_loader = data.DataLoader(X_data, batch_size, shuffle=True, num_workers=4)

Y_loader = data.DataLoader(Y_data, batch_size, shuffle=True, num_workers=4)
import time

from tqdm import tqdm_notebook as tqdm





resume = "../input/ganweights/checkpoint-29.pth"

start_epoch = 30

checkpoint = torch.load(resume)

G_X2Y.load_state_dict(checkpoint["G_X2Y"])

G_Y2X.load_state_dict(checkpoint["G_Y2X"])

D_X.load_state_dict(checkpoint["D_X"])

D_Y.load_state_dict(checkpoint["D_Y"])

optimizer_G_X2Y.load_state_dict(checkpoint["optimizer_G_X2Y"])

optimizer_G_Y2X.load_state_dict(checkpoint["optimizer_G_Y2X"])

optimizer_D_X.load_state_dict(checkpoint["optimizer_D_X"])

optimizer_D_Y.load_state_dict(checkpoint["optimizer_D_Y"])

print("loaded from", resume)



        

num_epochs = 10

for epoch in range(start_epoch, num_epochs+start_epoch):

    epoch_start_time = time.time()  # timer for entire epoch

    iter_data_time = time.time()

    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

    

    for epoch_iter, ((x, x_label), (y, y_label)) in tqdm(enumerate(zip(X_loader, Y_loader)), total=len(X_loader), desc="epoch {}".format(epoch)):

        t1 = time.time()



        x = x.cuda()

        x_label = x_label.cuda()

        y = y.cuda()

        y_label = y_label.cuda()

        

        fake_y = G_X2Y(x)

        rec_x = G_Y2X(fake_y, x_label)

        fake_x = G_Y2X(y, x_label)

        rec_y = G_X2Y(fake_x)



        pr = (D_Y(y))

#         pf = (D_Y(fake_y))

        pf = (D_Y(fake_y.detach()))

        sr = (D_X(x, x_label))

#         sf = (D_X(fake_x, x_label))   

        sf = (D_X(fake_x.detach(), x_label))   

        sw = (D_X(x, y_label))

    

    

        optimizer_D_X.zero_grad()

        optimizer_D_Y.zero_grad()

        gradients = torch.ones(batch_size,1).cuda()

        

        L_DY = -(torch.log(pr) + torch.log(1-pf))

        L_DY.backward(gradients, retain_graph=True)

        L_DX = -(torch.log(sr) + (torch.log(1-sf) + torch.log(1-sw)) / 2)

        L_DX.backward(gradients, retain_graph=True)

        optimizer_D_X.step()

        optimizer_D_Y.step()

        

    

        lambda1 = 0.1

        lambda2 = 0.1

#         gradients = torch.ones_like(L_GX2Y).cuda()

        optimizer_G_X2Y.zero_grad()

        optimizer_G_Y2X.zero_grad()

        L_c = lambda1 * torch.norm(x - rec_x, p = 1) + lambda2 * torch.norm(y - rec_y, p = 1)

        L_GX2Y = torch.log(pf) + L_c

#         print(torch.norm(x - fake_x, p = 1))

        L_GX2Y.backward(gradients, retain_graph=True)

        L_GY2X = torch.log(sf) + L_c

        L_GY2X.backward(gradients, retain_graph=False)

        

        optimizer_G_X2Y.step()

        optimizer_G_Y2X.step()



        if epoch_iter % 500 == 0:

#             print("{}/{} iter : {}".format(i,epoch, time.time() - iter_data_time))

            showRes(x,y, fake_x, fake_y, rec_x, rec_y)

            print("x_label:{}\ny_label:{}".format(x_label[0].detach().cpu().numpy(), y_label[0].detach().cpu().numpy()))

            print("L_DX:{:.6f} L_DY:{:.6f}".format(torch.mean(L_DX).detach().cpu().numpy().item(), torch.mean(L_DY).detach().cpu().numpy().item()))

            print("L_GX2Y:{:.6f} L_GY2X:{:.6f}".format(torch.mean(L_GX2Y).detach().cpu().numpy().item(), torch.mean(L_GY2X).detach().cpu().numpy().item()))

            

            torch.cuda.empty_cache()

        

    torch.save(

      {

          "G_X2Y":G_X2Y.state_dict(),

          "G_Y2X":G_Y2X.state_dict(),

          "D_X":D_X.state_dict(),

          "D_Y":D_Y.state_dict(),

          "optimizer_G_X2Y":optimizer_G_X2Y.state_dict(),

          "optimizer_G_Y2X":optimizer_G_Y2X.state_dict(),

          "optimizer_D_X":optimizer_D_X.state_dict(),

          "optimizer_D_Y":optimizer_D_Y.state_dict(),

      },

      './checkpoint-{}.pth'.format(epoch)

    )
['Bald', 'Smiling', 'Blond_Hair', 'Eyeglasses', 'Pale_Skin' , 'Wearing_Hat']