%matplotlib inline
from collections import OrderedDict

from itertools import chain

import io

import math

from multiprocessing import cpu_count

from pathlib import Path

from pdb import set_trace

import time

from threading import Thread

from xml.etree import ElementTree

import zipfile



# General utils

from allennlp.training.learning_rate_schedulers import CosineWithRestarts

from imageio import imread

from joblib import Parallel, delayed

import numpy as np

import pandas as pd

import PIL.Image

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from scipy.stats import truncnorm

from sklearn.preprocessing import LabelEncoder



# PyTorch

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as dset

import torchvision.transforms as transforms

from torchvision.utils import make_grid, save_image
HOUR = 3600



class Watchdog:

    def __init__(self, max_seconds=8.5 * HOUR):

        self.start = time.time()

        self.deadline = max_seconds

    

    @property

    def timeout(self):

        return self.elapsed >= self.deadline

       

    @property

    def elapsed(self):

        return time.time() - self.start



wd = Watchdog()



SEED = 1

np.random.seed(SEED)

torch.manual_seed(SEED)

ANNOTS = Path.cwd().parent/'input'/'annotation'/'Annotation'

IMAGES = Path.cwd().parent/'input'/'all-dogs'/'all-dogs'



try:

    # make sure we patch printing function only once

    patched

except NameError:

    patched = True

    __print__ = print

    def print(message):

        import os

        from datetime import datetime

        log_message = datetime.now().strftime(f'[Kernel][%Y-%m-%d %H:%M:%S] {message}')

        os.system(f'echo \"{log_message}\"')

        __print__(message)

        

class VisualStyle:

    """Convenience wrapper on top of matplotlib config."""



    def __init__(self, config, default=None):

        if default is None:

            default = plt.rcParams

        self.default = default.copy()

        self.config = config



    def replace(self):

        plt.rcParams = self.config



    def override(self, extra=None):

        plt.rcParams.update(self.config)

        if extra is not None:

            plt.rcParams.update(extra)



    def restore(self):

        plt.rcParams = self.default



    def __enter__(self):

        self.override()

        return self



    def __exit__(self, exc_type, exc_val, exc_tb):

        self.restore()





class NotebookStyle(VisualStyle):

    def __init__(self):

        super().__init__({

            'figure.figsize': (11, 8),

            'axes.titlesize': 20,

            'axes.labelsize': 18,

            'xtick.labelsize': 14,

            'ytick.labelsize': 14,

            'font.size': 16

        })



NotebookStyle().override()

        

print(f'Annotations: {ANNOTS}')

print(f'Images: {IMAGES}')
def parse_annotation(path):

    root = ElementTree.parse(path).getroot()

    size = [int(root.find(f'size/{leaf}').text) 

            for leaf in ('width', 'height')] 

    bbox = [int(root.find(f'object/bndbox/{leaf}').text) 

            for leaf in ('xmin', 'ymin', 'xmax', 'ymax')]

    breed = path.parent.name.split('-')[-1]

    return {'path': str(path), 'name': path.name, 

            'breed': breed, 'size': size, 'bbox': bbox}



def enrich_with_image_paths(annotations, images_directory):

    image_files = {x.stem: x for x in images_directory.iterdir()}

    enriched_data = []

    for annot in annotations:

        image_path = image_files.get(annot['name'], None)

        if image_path is None:

            print('Warning: image not found for annotation entry: %s.' % annot['path'])

            continue

        annot['image'] = str(image_path)

        enriched_data.append(annot)

    return enriched_data



def load_annotations():

    return enrich_with_image_paths([

        parse_annotation(path) 

        for directory in ANNOTS.iterdir() 

        for path in directory.iterdir()

    ], IMAGES)



def dog(annot):

    img = imread(annot['image'])

    xmin, ymin, xmax, ymax = annot['bbox']

    cropped = img[ymin:ymax, xmin:xmax]    

    return cropped



def chunks(seq, chunk_size=10):

    n = len(seq)

    n_chunks = n // chunk_size + int((n % chunk_size) != 0)

    for i in range(n_chunks):

        yield seq[i*chunk_size:(i+1)*chunk_size]

        

def resize(image, new_size):

    return np.array(PIL.Image.fromarray(image).resize(new_size))



def parallel(func, sequence, func_args=None, n_jobs=None):

    with Parallel(n_jobs=n_jobs or cpu_count()) as p:

        func_args = func_args or {}

        results = p(delayed(func)(item, **func_args) for item in sequence)

    return results



def load_single_image(annot, size):

    cropped = dog(annot)

    resized = resize(cropped, size)

    return resized



def load_dogs_images(annots, size=(64, 64)):

    return np.stack(parallel(load_single_image, annots, func_args={'size': size}))



def as_pil_list(dataset):

    return [PIL.Image.fromarray(image, 'RGB') for image in dataset]
print('Reading dogs images and annotations.')

annots = load_annotations()

print(f'Total number of examples: {len(annots)}.')

dogs = load_dogs_images(annots, (128, 128))

assert len(dogs) == len(annots)

print(f'Dogs dataset shape: {dogs.shape}.')

pils = as_pil_list(dogs)

print(f'Numbers of PIL images: {len(pils)}')

del dogs, annots
def show_pil(img, *imgs, n_rows=4):

    imgs = [img] + list(imgs)

    n_cols = len(imgs) // n_rows

    f, axes = plt.subplots(n_rows, n_cols)

    for img, ax in zip(imgs, axes.flat): 

        ax.imshow(img)

        ax.axis('off')

    f.subplots_adjust(wspace=0, hspace=0)
# show_pil(*pils[:16])
class PILDataset:

    def __init__(self, pil_images, transform=None):

        self.pil_images = pil_images

        self.tr = transform or (lambda x: x)

    def __getitem__(self, i):

        if isinstance(i, int): return self.tr(self.pil_images[i])

        elif isinstance(i, (list, np.ndarray)): return [self.tr(self.pil_images[ii]) for ii in i]

        elif isinstance(i, slice): return [self.tr(img) for img in self.pil_images[i]]

        raise TypeError(f'unknown index type: {type(i)}')

    def __len__(self):

        return len(self.pil_images)



class RandomCropOfFive:

    def __init__(self, size):

        self.five_crop = transforms.FiveCrop(size)

    def __call__(self, x):

        [idx] = np.random.randint(0, 4, 1)

        cropped = self.five_crop(x)[idx]

        return cropped

    

def show_tensor(t, n_rows=4, denorm=False):

    if denorm: t = (255 * (t + 1)/2)

    canvas = make_grid(t).numpy().transpose(1, 2, 0).astype(np.uint8)

    f, ax = plt.subplots(1, 1)

    ax.imshow(canvas)

    ax.axis('off')
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)

        

class Reshape(nn.Module):

    def __init__(self, shape):

        super().__init__()

        self.shape = shape

    def forward(self, x):

        return x.view(x.size(0), *self.shape)

    def __repr__(self):

        return f'{self.__class__.__name__}(shape={self.shape})'

    

class Flatten(nn.Module):

    def forward(self, x):

        return x.view(x.size(0), -1)



class Block(nn.Module):

    def __init__(

        self, ni, no,

        k=3, s=1, p=1, 

        scale=None, avgpool=None,

        ws=False, gain_ws=2, bn=False, pn=False, 

        activ=nn.LeakyReLU

    ):

        super().__init__()

        module = nn.Conv2d(ni, no, kernel_size=k, stride=s, padding=p, bias=not bn)

        if ws: module = WeightScaler(module, gain=gain_ws) 

        layers = [module]

        if scale is not None:

            layers.insert(0, nn.UpsamplingNearest2d(scale_factor=scale))

        if bn: layers.append(nn.BatchNorm2d(no))

        if activ is not None:

            if activ == nn.ReLU:

                layers.append(activ(True))

            elif activ == nn.LeakyReLU:

                layers.append(activ(negative_slope=0.2, inplace=True))

            elif activ == nn.PReLU:

                layers.append(activ(num_parameters=1))

            else:

                raise NotImplementedError('unknown activation layer')

        if pn: layers.append(PixelwiseNorm())

        if avgpool is not None:

            layers.append(nn.AvgPool2d(avgpool))

        self.block = nn.ModuleList(layers)

        

    def forward(self, x):

        for layer in self.block:

            x = layer(x)

        return x

    

class SpectralNorm(nn.Module):

    def __init__(self, module):

        super().__init__()

        self.module = nn.utils.spectral_norm(module)

    def forward(self, x):

        return self.module(x)



class PixelwiseNorm(nn.Module):

    def __init__(self, alpha=1e-8):

        super().__init__()

        self.alpha = alpha

    def forward(self, x):

        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.alpha).sqrt()

        y = x / y

        return y

    

class MinibatchStdDev(nn.Module):

    def __init__(self, alpha=1e-8):

        super().__init__()

        self.alpha = alpha

    def forward(self, x):

        batch_size, _, height, width = x.shape

        y = x - x.mean(dim=0, keepdim=True)

        y = y.pow(2.).mean(dim=0, keepdim=False).add(self.alpha).sqrt()

        y = y.mean().view(1, 1, 1, 1)

        y = y.repeat(batch_size, 1, height, width)

        y = torch.cat([x, y], 1)

        return y

    

class WeightScaler(nn.Module):

    def __init__(self, wrapped, gain=2):

        super().__init__()

        self.wrapped = wrapped

        self.gain = gain

        self.scale = (self.gain / wrapped.weight[0].numel()) ** 0.5

    def forward(self, x):

        return self.wrapped(x) * self.scale

    def __repr__(self):

        return f'{self.__class__.__name__}({self.wrapped}, c={self.gain})'

    

class Mixup:

    def __init__(self, alpha=0.2):

        self.alpha = alpha

    def __call__(self, b1, b2): 

        assert b1.size(0) == b2.size(0)

        lam = np.random.beta(self.alpha, self.alpha, size=b1.size(0))

        lam = torch.from_numpy(lam).float().to(b1.device)

        lam = lam.view(-1, 1, 1, 1)

        return lam*b1 + (1 - lam)*b2
class Generator(nn.Module):

    def __init__(self, latent_dim):

        super().__init__()

        self.to_rgbs = self.create_to_rgb_blocks()

        self.base = nn.Sequential(

            PixelwiseNorm(),

            nn.Linear(latent_dim, 4096),

            nn.LeakyReLU(0.2, True),

            PixelwiseNorm(),

            Reshape((256, 4, 4)),

            nn.Conv2d(256, 512, 3, 1, 1),

            nn.LeakyReLU(0.2, True),

            PixelwiseNorm()

        )  # output shape: B x 512 x 1 x 1

        self.conv_1 = nn.ModuleList([

            Block(512, 256, k=3, s=1, p=1, pn=True, scale=2),

            Block(256, 128, k=3, s=1, p=1, pn=True, scale=2),

            Block(128,  64, k=3, s=1, p=1, pn=True, scale=2),

            Block( 64,  32, k=3, s=1, p=1, pn=True, scale=2)

        ])

        self.conv_2 = nn.ModuleList([

            Block(256, 256, k=3, s=1, p=1, pn=True),

            Block(128, 128, k=3, s=1, p=1, pn=True),

            Block( 64,  64, k=3, s=1, p=1, pn=True),

            Block( 32,  32, k=3, s=1, p=1, pn=True)            

        ])

        self.tanh = nn.Tanh()

        self.apply(weights_init)

        

    def create_to_rgb_blocks(self):

        def to_rgb(n): return nn.Conv2d(n, 3, kernel_size=3, stride=1, padding=1)

        layers = [to_rgb(x) for x in (512, 256, 128, 64, 32)]

        return nn.ModuleList(layers)

    

    def forward(self, z, block=0, alpha=0.0):

        x = self.base(z)

        if block == 0:

            return self.tanh(self.to_rgbs[0](x))

        prev = None

        for i in range(block):

            prev = x

            x = self.conv_1[i](x)

            x = self.conv_2[i](x)

        # upper branch

        a = self.to_rgbs[block](x)

        # lower branch

        b = F.upsample_nearest(self.to_rgbs[block-1](prev), scale_factor=2)

        blended = alpha*a + (1-alpha)*b

        return self.tanh(blended)
class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.from_rgbs = self.create_from_rgb_blocks()

        self.conv_1 = nn.ModuleList([

            Block( 32,  32, k=3, s=1, p=1),

            Block( 64,  64, k=3, s=1, p=1),

            Block(128, 128, k=3, s=1, p=1),

            Block(256, 256, k=3, s=1, p=1),

        ])

        self.conv_2 = nn.ModuleList([

            Block( 32,  64, k=3, s=1, p=1, avgpool=2),

            Block( 64, 128, k=3, s=1, p=1, avgpool=2),

            Block(128, 256, k=3, s=1, p=1, avgpool=2),

            Block(256, 512, k=3, s=1, p=1, avgpool=2),

        ])

        self.top = nn.Sequential(

            MinibatchStdDev(),

            nn.Conv2d(513, 512, 3, 1, 1),

            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 4, 2, 0),

            nn.LeakyReLU(0.2, True),

            Flatten(),

            nn.Linear(512, 1),

            nn.Sigmoid()

        )

    

    def create_from_rgb_blocks(self):

        def from_rgb(n): return nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1)

        layers = [from_rgb(x) for x in (32, 64, 128, 256, 512)]

        return nn.ModuleList(layers)

    

    def forward(self, x, block=0, alpha=0.0):

        act = lambda value: F.leaky_relu(value, 0.2)

        

        if block == 0:

            # input: 4x4

            x = F.leaky_relu(self.from_rgbs[-1](x), 0.2)

            x = self.top(x)

            return x

        

        elif block == 1:

            # input: 8x8

            

            # upper branch

            a = act(self.from_rgbs[-2](x))

            a = self.conv_1[-1](a)

            a = self.conv_2[-1](a)

            

            # lower branch

            b = act(self.from_rgbs[-1](F.avg_pool2d(x, 2)))

            

            blended = alpha*a + (1-alpha)*b

            x = self.top(blended)

            return x

        

        elif block == 2:

            # input: 16x16

            

            # upper branch

            a = act(self.from_rgbs[-3](x))

            a = self.conv_1[-2](a)

            a = self.conv_2[-2](a)

            

            # lower branch

            b = act(self.from_rgbs[-2](F.avg_pool2d(x, 2)))

            

            blended = alpha*a + (1-alpha)*b

            

            x = self.conv_1[-1](blended)

            x = self.conv_2[-1](x)

            x = self.top(x)

            return x

        

        elif block == 3:

            # input: 32x32

            

            # upper branch

            a = act(self.from_rgbs[-4](x))

            a = self.conv_1[-3](a)

            a = self.conv_2[-3](a)

            

            # lower branch

            b = act(self.from_rgbs[-3](F.avg_pool2d(x, 2)))

            

            blended = alpha*a + (1-alpha)*b

            

            x = self.conv_1[-2](blended)

            x = self.conv_2[-2](x)

            x = self.conv_1[-1](x)

            x = self.conv_2[-1](x)

            x = self.top(x)

            return x

        

        elif block == 4:

            # input: 64x64

            

            # upper branch

            a = act(self.from_rgbs[-5](x))

            a = self.conv_1[-4](a)

            a = self.conv_2[-4](a)

            

            # lower branch

            b = act(self.from_rgbs[-4](F.avg_pool2d(x, 2)))

            

            blended = alpha*a + (1-alpha)*b

            

            x = self.conv_1[-3](blended)

            x = self.conv_2[-3](x)

            x = self.conv_1[-2](x)

            x = self.conv_2[-2](x)

            x = self.conv_1[-1](x)

            x = self.conv_2[-1](x)

            x = self.top(x)

            return x
# d = Critic()

# img = torch.ones((10, 3, 4, 4))

# out = d(img, block=0, alpha=0.5)

# out.shape
# d = Critic()

# img = torch.ones((10, 3, 8, 8))

# out = d(img, block=1, alpha=0.5)

# out.shape
# d = Critic()

# img = torch.ones((10, 3, 16, 16))

# out = d(img, block=2, alpha=0.5)

# out.shape
# d = Critic()

# img = torch.ones((10, 3, 32, 32))

# out = d(img, block=3, alpha=0.5)

# out.shape
# d = Critic()

# img = torch.ones((10, 3, 64, 64))

# out = d(img, block=4, alpha=0.5)

# out.shape
# g = Generator(128)

# z = torch.randn((10, 128))

# x = g(z)

# x.shape
bs = 16

nz = 128

lr_d = 0.0001

lr_g = 0.0001

beta_1 = 0.5

beta_2 = 0.999

mixup = Mixup(0.2)



dataset = PILDataset(pils, transform=transforms.Compose([

    transforms.Resize(70),

    RandomCropOfFive(64),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(3),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

]))



netD = Critic().cuda()

optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta_1, beta_2))

schedD = CosineWithRestarts(optimizerD, eta_min=lr_d*0.1, t_initial=1000, t_mul=math.sqrt(2))

                        

netG = Generator(nz).cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta_1, beta_2))

schedG = CosineWithRestarts(optimizerG, eta_min=lr_g*0.1, t_initial=1000, t_mul=math.sqrt(2))
# show_tensor(torch.stack(dataset[np.random.randint(0, len(dataset), 64)]), denorm=True)
def truncated_normal(size, threshold=1):

    return truncnorm.rvs(-threshold, threshold, size=size)



def hypersphere(z, radius=1):

    return z * radius / z.norm(p=2, dim=1, keepdim=True)



def sample(dataset, batch_size):

    idx = np.random.randint(0, len(dataset), batch_size)

    return torch.stack(dataset[idx]).cuda()



def smooth_positive(labels):

    jitter = torch.from_numpy(np.random.uniform(0.05, 0.1, len(labels))).float().to(labels.device)

    jitter = jitter.view(labels.size())

    return (labels - jitter)
# def train(bs=32):

def train():

    print('Starting training loop...')

    epoch = 0

    real_label = 1

    fake_label = 0

    loss_fn = nn.BCELoss()

    n = len(dataset)

    # n_batches = n // bs

    block = 0

    alpha_range = list(np.linspace(0, 1, 7500)) + list([1] * 7500)

    alpha_index = 0

    # batch_sizes = {0: 256, 1: 128, 2: 64, 3: 32, 4: 16}

    batch_sizes = {0: 512, 1: 256, 2: 128, 3: 32, 4: 16}

    

    while True:

        bs = batch_sizes[block]

        n_batches = n // bs

        idx1 = np.random.permutation(n)

        idx2 = np.random.permutation(n)

        

        for i in range(n_batches):

            

            if wd.timeout: return

            

            epoch += 1

            alpha_value = alpha_range[alpha_index]

            

            batch1 = torch.stack(dataset[idx1[i*bs:(i+1)*bs]]).float()

            batch2 = torch.stack(dataset[idx2[i*bs:(i+1)*bs]]).float()

            mixed = mixup(batch1, batch2)

            

            if block < 4:

                mixed = nn.functional.interpolate(mixed, size=(

                    (4, 4) if block == 0 else

                    (8, 8) if block == 1 else

                    (16, 16) if block == 2 else

                    (32, 32) if block == 3 else

                    (64, 64)

                ), mode='bilinear', align_corners=True).detach()

            

            netD.zero_grad()

            x_real = mixed.cuda()

            batch_size = x_real.size(0)

            labels = torch.full((batch_size, 1), real_label).cuda()

            labels = smooth_positive(labels) 

            output = netD(x_real, block=block, alpha=alpha_value).view(-1, 1)

            errD_real = loss_fn(output, labels)

            errD_real.backward()

            d_x = output.mean().item()



            # noise = torch.from_numpy(truncated_normal((batch_size, nz, 1, 1))).float().cuda()

            noise = torch.from_numpy(truncated_normal((batch_size, nz))).float().cuda()

            x_fake = netG(noise, block=block, alpha=alpha_value)

            labels.fill_(fake_label)

            output = netD(x_fake.detach(), block=block, alpha=alpha_value).view(-1, 1)

            errD_fake = loss_fn(output, labels)

            errD_fake.backward()

            d_g_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()



            netG.zero_grad()

            labels.fill_(real_label)

            output = netD(x_fake, block=block, alpha=alpha_value).view(-1, 1)

            errG = loss_fn(output, labels)

            errG.backward()

            d_g_z2 = output.mean().item()

            optimizerG.step()

            

            if epoch % 150 == 0:

                print(f'[{epoch:6d}] '

                      f'lr_d: {schedD.get_values()[0]:.6f}, '

                      f'lr_g: {schedG.get_values()[0]:.6f} | '

                      f'loss_d: {errD.item():.4f}, '

                      f'loss_g: {errG.item():.4f} | '

                      f'D(x): {d_x:.4f}, D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f} | '

                      f'no={block}, a={alpha_value:.4f}, bs={bs:d}, sz={list(mixed.shape[2:])}')

        

            if block == 4:

                schedD.step()

                schedG.step()

            

            alpha_index += 1

            

            if block == 4:

                # discard blending, use layers only

                alpha_index = -1

            elif alpha_index >= len(alpha_range):

                # start new cycle

                block += 1 

                alpha_index = 0
# train(bs=16)

train()
print('Final model images generation.')

print('Creating archive to write the images.')

arch = zipfile.ZipFile('images.zip', 'w')

img_no = 0

for batch in range(100):

    # t_noise = torch.from_numpy(truncated_normal((100, nz, 1, 1))).float().cuda()

    t_noise = torch.from_numpy(truncated_normal((100, nz))).float().cuda()

    images = netG(t_noise, block=4, alpha=1.0).detach().cpu()

    images = images.mul(0.5).add(0.5)

    images = (255 * images.numpy()).astype(np.uint8)

    images = images.transpose(0, 2, 3, 1)

    for image in images:

        buf = io.BytesIO()

        PIL.Image.fromarray(image).save(buf, format='png')

        buf.seek(0)

        arch.writestr(f'{img_no}.png', buf.getvalue())

        img_no += 1

arch.close()

print('Saving is done!')