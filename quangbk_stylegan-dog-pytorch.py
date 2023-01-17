import argparse

import random

import math



from tqdm import tqdm

import numpy as np

from PIL import Image



import torch

from torch import nn, optim

from torch.nn import functional as F

from torch.autograd import Variable, grad

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

import torchvision.utils as vutils

from torch.utils.data import Dataset, DataLoader

import os

from os.path import join

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt



def requires_grad(model, flag=True):

    for p in model.parameters():

        p.requires_grad = flag





def accumulate(model1, model2, decay=0.999):

    par1 = dict(model1.named_parameters())

    par2 = dict(model2.named_parameters())



    for k in par1.keys():

        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)





def adjust_lr(optimizer, lr):

    for group in optimizer.param_groups:

        mult = group.get('mult', 1)

        group['lr'] = lr * mult

class DogDataset(Dataset):

    def __init__(self, img_dir, annotations_dir, transform1=None, transform2=None):



        self.img_dir = img_dir

        self.img_names = os.listdir(img_dir)

        self.transform1 = transform1

        self.transform2 = transform2

        

        self.imgs = []

        for img_name in self.img_names:

            path = join(img_dir, img_name)

            img = datasets.folder.default_loader(path)

    

            # Crop image

            annotation_basename = os.path.splitext(os.path.basename(path))[0]

            annotation_dirname = next(dirname for dirname in os.listdir(annotations_dir) if dirname.startswith(annotation_basename.split('_')[0]))

            annotation_filename = os.path.join(annotations_dir, annotation_dirname, annotation_basename)

            tree = ET.parse(annotation_filename)

            root = tree.getroot()

            objects = root.findall('object')

            

            for o in objects:

                bndbox = o.find('bndbox')

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                bbox = (xmin, ymin, xmax, ymax)

                img_ = img.crop(bbox)

                # Some crop's are black. if crop is black then don't crop

                if np.mean(img_) != 0:

                    img = img_



                if self.transform1 is not None:

                    img = self.transform1(img)



                self.imgs.append(img)



    def __getitem__(self, index):

        img = self.imgs[index]

        

        if self.transform2 is not None:

            img = self.transform2(img)

        

        return img



    def __len__(self):

        return len(self.imgs)
def sample_data(PATH_IMG, batch_size, image_size=32):

    transform1 = transforms.Compose([transforms.Resize(image_size),

                                    transforms.CenterCrop(image_size)])



    # Data augmentation and converting to tensors

    random_transforms = [transforms.RandomRotation(degrees=5)]

    transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),

                                     transforms.RandomApply(random_transforms, p=0.3), 

                                     transforms.ToTensor(),

                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    train_dataset = DogDataset(img_dir='../input/all-dogs/all-dogs/',

                               annotations_dir='../input/annotation/Annotation/',

                               transform1=transform1,

                               transform2=transform2)



    train_loader = DataLoader(dataset=train_dataset,

                              batch_size=batch_size,

                              shuffle=True,

                              num_workers=4)



    return train_loader, train_dataset
import torch



from torch import nn

from torch.nn import init

from torch.nn import functional as F

from torch.autograd import Variable



from math import sqrt



import random



n_class_age = 6

n_repeat = 1



def init_linear(linear):

    init.xavier_normal(linear.weight)

    linear.bias.data.zero_()





def init_conv(conv, glu=True):

    init.kaiming_normal(conv.weight)

    if conv.bias is not None:

        conv.bias.data.zero_()





class EqualLR:

    def __init__(self, name):

        self.name = name



    def compute_weight(self, module):

        weight = getattr(module, self.name + '_orig')

        fan_in = weight.data.size(1) * weight.data[0][0].numel()



        return weight * sqrt(2 / fan_in)



    @staticmethod

    def apply(module, name):

        fn = EqualLR(name)



        weight = getattr(module, name)

        del module._parameters[name]

        module.register_parameter(name + '_orig', nn.Parameter(weight.data))

        module.register_forward_pre_hook(fn)



        return fn



    def __call__(self, module, input):

        weight = self.compute_weight(module)

        setattr(module, self.name, weight)





def equal_lr(module, name='weight'):

    EqualLR.apply(module, name)



    return module





class PixelNorm(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, input):

        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)





class Blur(nn.Module):

    def __init__(self):

        super().__init__()



        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)

        weight = weight.view(1, 1, 3, 3)

        weight = weight / weight.sum()

        self.register_buffer('weight', weight)



    def forward(self, input):

        return F.conv2d(

            input,

            self.weight.repeat(input.shape[1], 1, 1, 1),

            padding=1,

            groups=input.shape[1],

        )





class EqualConv2d(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__()



        conv = nn.Conv2d(*args, **kwargs)

        conv.weight.data.normal_()

        conv.bias.data.zero_()

        self.conv = equal_lr(conv)



    def forward(self, input):

        return self.conv(input)





class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim):

        super().__init__()



        linear = nn.Linear(in_dim, out_dim)

        linear.weight.data.normal_()

        linear.bias.data.zero_()



        self.linear = equal_lr(linear)



    def forward(self, input):

        return self.linear(input)





class ConvBlock(nn.Module):

    def __init__(

        self,

        in_channel,

        out_channel,

        kernel_size,

        padding,

        kernel_size2=None,

        padding2=None,

        pixel_norm=True,

        spectral_norm=False,

    ):

        super().__init__()



        pad1 = padding

        pad2 = padding

        if padding2 is not None:

            pad2 = padding2



        kernel1 = kernel_size

        kernel2 = kernel_size

        if kernel_size2 is not None:

            kernel2 = kernel_size2



        self.conv = nn.Sequential(

            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),

            nn.LeakyReLU(0.2),

            EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),

            nn.LeakyReLU(0.2),

        )



    def forward(self, input):

        out = self.conv(input)



        return out





class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channel, style_dim):

        super().__init__()



        self.norm = nn.InstanceNorm2d(in_channel)

        self.style = EqualLinear(style_dim, in_channel * 2)



        self.style.linear.bias.data[:in_channel] = 1

        self.style.linear.bias.data[in_channel:] = 0



    def forward(self, input, style):

        style = self.style(style).unsqueeze(2).unsqueeze(3)

        gamma, beta = style.chunk(2, 1)



        out = self.norm(input)

        out = gamma * out + beta



        return out





class NoiseInjection(nn.Module):

    def __init__(self, channel):

        super().__init__()



        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))



    def forward(self, image, noise):

        return image + self.weight * noise





class ConstantInput(nn.Module):

    def __init__(self, channel, size=8):

        super().__init__()



        self.input = nn.Parameter(torch.randn(1, channel, size, size))



    def forward(self, input):

        batch = input.shape[0]

        out = self.input.repeat(batch, 1, 1, 1)



        return out





class StyledConvBlock(nn.Module):

    def __init__(

        self,

        in_channel,

        out_channel,

        kernel_size=3,

        padding=1,

        style_dim=512,

        initial=False,

    ):

        super().__init__()



        if initial:

            self.conv1 = ConstantInput(in_channel)



        else:

            self.conv1 = EqualConv2d(

                in_channel, out_channel, kernel_size, padding=padding

            )



        self.noise1 = equal_lr(NoiseInjection(out_channel))

        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)

        self.lrelu1 = nn.LeakyReLU(0.2)



        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)

        self.noise2 = equal_lr(NoiseInjection(out_channel))

        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)

        self.lrelu2 = nn.LeakyReLU(0.2)



    def forward(self, input, style, noise):

        out = self.conv1(input)

        out = self.noise1(out, noise)

        out = self.adain1(out, style)

        out = self.lrelu1(out)



        out = self.conv2(out)

        out = self.noise2(out, noise)

        out = self.adain2(out, style)

        out = self.lrelu2(out)



        return out





class Generator(nn.Module):

    def __init__(self, code_dim):

        super().__init__()



        self.progression = nn.ModuleList(

            [

                StyledConvBlock(512, 512, 3, 1, initial=True),  #8

#                 StyledConvBlock(512, 512, 3, 1),

                StyledConvBlock(512, 256, 3, 1),                #16

                StyledConvBlock(256, 128, 3, 1),                #32

                StyledConvBlock(128, 64, 3, 1),                 #64

                StyledConvBlock(64, 32, 3, 1),

                StyledConvBlock(128, 64, 3, 1),

                StyledConvBlock(64, 32, 3, 1),

                StyledConvBlock(32, 16, 3, 1),

            ]

        )



        self.to_rgb = nn.ModuleList(

            [

                EqualConv2d(512, 3, 1),

#                 EqualConv2d(512, 3, 1),

                EqualConv2d(256, 3, 1),

                EqualConv2d(128, 3, 1),

                EqualConv2d(64, 3, 1),

                EqualConv2d(32, 3, 1),

                EqualConv2d(64, 3, 1),

                EqualConv2d(32, 3, 1),

                EqualConv2d(16, 3, 1),

            ]

        )



        # self.blur = Blur()



    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):

        out = noise[0]



        if len(style) < 2:

            inject_index = [len(self.progression) + 1]



        else:

            inject_index = random.sample(list(range(step)), len(style) - 1)



        crossover = 0



        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):

            if mixing_range == (-1, -1):

                if crossover < len(inject_index) and i > inject_index[crossover]:

                    crossover = min(crossover + 1, len(style))



                style_step = style[crossover]



            else:

                if mixing_range[0] <= i <= mixing_range[1]:

                    style_step = style[1]



                else:

                    style_step = style[0]



            if i > 0 and step > 0:

                upsample = F.interpolate(

                    out, scale_factor=2, mode='bilinear', align_corners=False

                )

                out = conv(upsample, style_step, noise[i])



            else:

                out = conv(out, style_step, noise[i])



            if i == step:

                out = to_rgb(out)



                if i > 0 and 0 <= alpha < 1:

                    skip_rgb = self.to_rgb[i - 1](upsample)

                    out = (1 - alpha) * skip_rgb + alpha * out



                break



        return out





class StyledGenerator(nn.Module):

    def __init__(self, code_dim=512, n_mlp=8):

        super().__init__()



        self.generator = Generator(code_dim)



        layers = [PixelNorm()]

        for i in range(n_mlp):

            layers.append(EqualLinear(code_dim, code_dim))

            layers.append(nn.LeakyReLU(0.2))



        self.style = nn.Sequential(*layers)



    def forward(

        self,

        input,

        noise=None,

        step=0,

        alpha=-1,

        mean_style=None,

        style_weight=0,

        mixing_range=(-1, -1),

    ):

        styles = []

        if type(input) not in (list, tuple):

            input = [input]



        for i in input:

            styles.append(self.style(i))



        batch = input[0].shape[0]



        if noise is None:

            noise = []



            for i in range(step + 1):

                size = 8 * 2 ** i

                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))



        if mean_style is not None:

            styles_norm = []



            for style in styles:

                styles_norm.append(mean_style + style_weight * (style - mean_style))



            styles = styles_norm



        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)



    def mean_style(self, input):

        style = self.style(input).mean(0, keepdim=True)



        return style





class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()



        self.progression = nn.ModuleList(

            [

                ConvBlock(16, 32, 3, 1),

                ConvBlock(32, 64, 3, 1),

                ConvBlock(64, 128, 3, 1),

                ConvBlock(32, 64, 3, 1),

                ConvBlock(64, 128, 3, 1),

                ConvBlock(128, 128, 3, 1),

                ConvBlock(128, 256, 3, 1),

#                 ConvBlock(512, 512, 3, 1),

                ConvBlock(257, 512, 3, 1, 8, 0),

            ]

        )



        self.from_rgb_ = nn.ModuleList(

            [

                EqualConv2d(3, 16, 1),

                EqualConv2d(3, 32, 1),

                EqualConv2d(3, 64, 1),

                EqualConv2d(3, 32, 1),

                EqualConv2d(3, 64, 1),

                EqualConv2d(3, 128, 1),

                EqualConv2d(3, 128, 1),

#                 EqualConv2d(3+n_class_age, 512, 1),

                EqualConv2d(3, 256, 1),

            ]

        )



        self.n_layer = len(self.progression)



        self.linear = EqualLinear(512, 1)



    def forward(self, input, step=0, alpha=-1):

        

        img_size = input.shape[2]



        for i in range(step, -1, -1):

            index = self.n_layer - i - 1



            if i == step:

                input_ = input

                out = self.from_rgb_[index](input_)



            if i == 0:

                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)

                mean_std = out_std.mean()

                mean_std = mean_std.expand(out.size(0), 1, 8, 8)

                out = torch.cat([out, mean_std], 1)



            out = self.progression[index](out)



            if i > 0:

                # out = F.avg_pool2d(out, 2)

                out = F.interpolate(

                    out, scale_factor=0.5, mode='bilinear', align_corners=False

                )



                if i == step and 0 <= alpha < 1:

                    skip_rgb = F.interpolate(

                        input_, scale_factor=0.5, mode='bilinear', align_corners=False

                    )

                    skip_rgb = self.from_rgb_[index + 1](skip_rgb)



                    out = (1 - alpha) * skip_rgb + alpha * out



        out = out.squeeze(2).squeeze(2)

#         print(input.size(), out.size(), step)

        out = self.linear(out)



        return out
code_size = 512

batch_size = 16

n_critic = 1

Is_Show = False

class Args:

    n_gpu = 4

#     phase = 600_000

    phase = 150_000

    lr = 0.001

    init_size = 8

    max_size = 64

    mixing = False

    loss = 'wgan-gp'

    data = 'folder'

    path = '/home/quang/working/Dog_kaggle/data/dog-resize/all-dogs/'

    sched = None

    

args = Args()

generator = nn.DataParallel(StyledGenerator(code_size)).cuda()

discriminator = nn.DataParallel(Discriminator()).cuda()



class_loss = nn.CrossEntropyLoss()

g_optimizer = optim.Adam(

    generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)

)    

g_optimizer.add_param_group(

    {

        'params': generator.module.style.parameters(),

        'lr': args.lr * 0.01,

        'mult': 0.01,

    }

)



d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))



if args.sched:

    args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

    args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}



else:

    args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

    args.batch = {4: 32, 8: 32, 16: 32, 32: 32, 64: 32, 128: 16, 256: 8}



args.gen_sample = {512: (8, 4), 1024: (4, 2)}



args.batch_default = 32
step = int(math.log2(args.init_size)) - 3

resolution = 8 * 2 ** step

loader, dog_dataset = sample_data(

    args.path, args.batch.get(resolution, args.batch_default), resolution

)

data_loader = iter(loader)
adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))

adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))



pbar = tqdm(range(60_000))

# pbar = tqdm(range(100))



requires_grad(generator, False)

requires_grad(discriminator, True)



disc_loss_val = 0

gen_loss_val = 0

grad_loss_val = 0



alpha = 0

used_sample = 0



n_repeat = 2



print ('Resolution: ', resolution, '|Step: ', step, '|Batch_size: ', args.batch.get(resolution, args.batch_default), ' |Generator lr: ', 

      g_optimizer.state_dict()['param_groups'][0]['lr'], ' |Style lr: ', g_optimizer.state_dict()['param_groups'][1]['lr'])

for i in pbar:

    discriminator.zero_grad()



    alpha = min(1, 1 / args.phase * (used_sample + 1))



    if used_sample > args.phase * 2 and step < (int(math.log2(args.max_size)) - 3):

        step += 1



        if step > int(math.log2(args.max_size)) - 3:

            step = int(math.log2(args.max_size)) - 3



        else:

            alpha = 0

            used_sample = 0



        resolution = 8 * 2 ** step

        del loader

        del dog_dataset

        loader, dog_dataset = sample_data(

            args.path, args.batch.get(resolution, args.batch_default), resolution

        )

        data_loader = iter(loader)



        adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))

        adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        print ('Resolution: ', resolution, '|Step: ', step, '|Batch_size: ', args.batch.get(resolution, args.batch_default), ' |Generator lr: ', 

              g_optimizer.state_dict()['param_groups'][0]['lr'], ' |Style lr: ', g_optimizer.state_dict()['param_groups'][1]['lr'])



    try:

        real_image = next(data_loader)



    except (OSError, StopIteration):

        data_loader = iter(loader)

        real_image = next(data_loader)



    used_sample += real_image.shape[0]



    b_size = real_image.size(0)

    real_image = real_image.cuda()



    if args.loss == 'wgan-gp':

        real_predict = discriminator(real_image, step=step, alpha=alpha)

        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()

        (-real_predict).backward()



    elif args.loss == 'r1':

        real_image.requires_grad = True

        real_predict = discriminator(real_image, step=step, alpha=alpha)

        real_predict = F.softplus(-real_predict).mean()

        real_predict.backward(retain_graph=True)



        grad_real = grad(

            outputs=real_predict.sum(), inputs=real_image, create_graph=True

        )[0]

        grad_penalty = (

            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2

        ).mean()

        grad_penalty = 10 / 2 * grad_penalty

        grad_penalty.backward()

        grad_loss_val = grad_penalty.item()



    if args.mixing and random.random() < 0.9 and False:

        gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(

            4, b_size, code_size, device='cuda'

        ).chunk(4, 0)

        gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]

        gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]



    else:

        gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(2, 0)            

        gen_in1 = gen_in1.squeeze(0)

        gen_in2 = gen_in2.squeeze(0)



    fake_image = generator(gen_in1, step=step, alpha=alpha)

    fake_predict = discriminator(fake_image, step=step, alpha=alpha)



    if args.loss == 'wgan-gp':

        fake_predict = fake_predict.mean()

        fake_predict.backward()



        eps = torch.rand(b_size, 1, 1, 1).cuda()

        x_hat = eps * real_image.data + (1 - eps) * fake_image.data

        x_hat.requires_grad = True

        hat_predict = discriminator(x_hat, step=step, alpha=alpha)

        grad_x_hat = grad(

            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True

        )[0]

        grad_penalty = (

            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2

        ).mean()

        grad_penalty = 10 * grad_penalty

        grad_penalty.backward()

        grad_loss_val = grad_penalty.item()

        disc_loss_val = (real_predict - fake_predict).item()



    elif args.loss == 'r1':

        fake_predict = F.softplus(fake_predict).mean()

        fake_predict.backward()

        disc_loss_val = (real_predict + fake_predict).item()



    d_optimizer.step()



    if (i + 1) % n_critic == 0:

        generator.zero_grad()



        requires_grad(generator, True)

        requires_grad(discriminator, False)



        fake_image = generator(gen_in2, step=step, alpha=alpha)



        predict = discriminator(fake_image, step=step, alpha=alpha)



        if args.loss == 'wgan-gp':

            loss = -predict.mean()



        elif args.loss == 'r1':

            loss = F.softplus(-predict).mean()



        gen_loss_val = loss.item()



        loss.backward()

        g_optimizer.step()



        requires_grad(generator, False)

        requires_grad(discriminator, True)



    if (i + 1) % 500 == 0 and Is_Show:

        images = []



        gen_i, gen_j = args.gen_sample.get(resolution, (5, 10))

        random_z = torch.randn(gen_j, code_size).cuda()

        with torch.no_grad():

            for age_code in range(gen_i):

                gen_test = torch.randn(gen_j, code_size).cuda()

                images.append(generator(gen_test, step=step, alpha=alpha))

       

        gen_image_temp = vutils.make_grid(torch.cat(images, 0), nrow=10, padding=2, normalize=True)

        gen_image_temp = gen_image_temp.cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(16,6))

        plt.imshow(gen_image_temp)

        plt.show()



    state_msg = (

        f'Size: {8 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'

        f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'

    )



    pbar.set_description(state_msg)
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL = False

if LOCAL:

    OUTPUT_PATH = './output_images'

else:

    OUTPUT_PATH = '../output_images'

if not os.path.exists(OUTPUT_PATH):

    os.mkdir(OUTPUT_PATH)

    

im_batch_size = 50

n_images=10000

# n_images = 5

with torch.no_grad():

    for i_batch in range(0, n_images, im_batch_size):

        gen_test = torch.randn(im_batch_size, code_size, device=device)

    #     gen_images = (netG(gen_z)+1.)/2.

        gen_images = (generator(gen_test, step=step, alpha=alpha) + 1.)/2.

        images = gen_images.to("cpu").clone().detach()

        images = images.numpy().transpose(0, 2, 3, 1)

        for i_image in range(gen_images.size(0)):

            save_image(gen_images[i_image, :, :, :], os.path.join(OUTPUT_PATH, f'image_{i_batch+i_image:05d}.png'), normalize=True)



import shutil

shutil.make_archive('images', 'zip', OUTPUT_PATH)