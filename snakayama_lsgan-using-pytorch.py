# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# basic modules

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import cv2

import matplotlib.animation as animation

from IPython.display import HTML

import imgaug

import random

from imgaug import augmenters as iaa

from imgaug import parameters as iap

from PIL import Image



#keras modules

from keras.preprocessing.image import img_to_array, load_img



#Pytorch modules

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.optim as optim

import torch.utils.data

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils

from torchvision.utils import save_image

from torch.nn.utils import spectral_norm



#Crop images using bounding box

import xml.etree.ElementTree as ET

import glob
img_list =os.listdir("../input/all-dogs/all-dogs/")
len(img_list)
root_images="../input/all-dogs/all-dogs/"

root_annots="../input/annotation/Annotation/"

croped_images="../croped_images/"
all_images=os.listdir("../input/all-dogs/all-dogs/")

print(f"Total images : {len(all_images)}")



breeds = glob.glob('../input/annotation/Annotation/*')

annotation=[]

for b in breeds:

    annotation+=glob.glob(b+"/*")

print(f"Total annotation : {len(annotation)}")



breed_map={}

for annot in annotation:

    breed=annot.split("/")[-2]

    index=breed.split("-")[0]

    breed_map.setdefault(index,breed)

    

print(f"Total Breeds : {len(breed_map)}")
def bounding_box(image):

    bpath=root_annots+str(breed_map[image.split("_")[0]])+"/"+str(image.split(".")[0])

    tree = ET.parse(bpath)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        

    return (xmin,ymin,xmax,ymax)
if not os.path.exists(croped_images):

    os.mkdir(croped_images)
for i,image in enumerate(all_images):

    #print(image)

    bbox=bounding_box(image)

    im=Image.open(os.path.join(root_images,image))

    im=im.crop(bbox)

    im.save(croped_images + image, quality=95)
len(os.listdir("../croped_images/"))
temp_img = load_img('../croped_images/n02085620_3423.jpg')

temp_img_array  = img_to_array(temp_img)
temp_img
temp_img_array.shape
sns.set_style("white")

count = 1

plt.figure(figsize=[20, 20])

for img_name in img_list[:15]:

    #print("../input/all-dogs/all-dogs/%s.jpg" % img_name)

    img = cv2.imread("../croped_images/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    count += 1

    

plt.show()
manualSeed = random.randint(1000, 10000)

print("Random Seed: ", manualSeed)

random.seed(manualSeed)

torch.manual_seed(manualSeed)
if not os.path.exists('../result_images/'):

    os.mkdir('../result_images/')
# Root directory for dataset

dataroot = "../"

image_size = 64

nc = 3

nz = 128

ngf = 64

ndf = 64

num_epochs = 200

lr = 0.0001

beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.

ngpu = 1



# Initial_setting

workers = 2

batch_size=64  

nz = 100

nch_g = 64

nch_d = 64

n_epoch = 200

lr = 0.0002

beta1 = 0.5

outf = '../result_images'

display_interval = 100

save_fake_image_interval = 1500

plt.rcParams['figure.figsize'] = 10, 6
# Dataset Creator

rand_aff = random.uniform(3.0, 15.0)

rand_flip = random.uniform(0.3, 1.0)

rand_trans = random.uniform(0.3, 0.7)

rand_contr = random.uniform(0.2, 0.9)

random_transforms = [transforms.ColorJitter(contrast=rand_contr), transforms.RandomAffine(degrees=rand_aff)]



dataset = dset.ImageFolder(root=dataroot,

                          transform=transforms.Compose([

                          transforms.RandomResizedCrop(64, scale=(0.9, 1.0), ratio=(1., 1.)),

                          transforms.RandomHorizontalFlip(),

                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),

                          transforms.ToTensor(),

                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                      ]))  



# Create the dataloader

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,

                                         shuffle=True, num_workers=workers)



# Decide which device we want to run on

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
real_batch = next(iter(dataloader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:            

        m.weight.data.normal_(0.0, 0.02)

        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:        

        m.weight.data.normal_(0.0, 0.02)

        m.bias.data.fill_(0)

    elif classname.find('BatchNorm') != -1:    

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)
class Generator(nn.Module):

    def __init__(self, nz=100, nch_g=64, nch=3):

        super(Generator, self).__init__()

        self.layers = nn.ModuleDict({

            'layer0': nn.Sequential(

                spectral_norm(nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0)),     

                nn.BatchNorm2d(nch_g * 8),                      

                nn.ReLU()                                       

            ),  # (100, 1, 1) -> (512, 4, 4)

            'layer1': nn.Sequential(

                spectral_norm(nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1)),

                nn.BatchNorm2d(nch_g * 4),

                nn.ReLU()

            ),  # (512, 4, 4) -> (256, 8, 8)

            'layer2': nn.Sequential(

                spectral_norm(nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1)),

                nn.BatchNorm2d(nch_g * 2),

                nn.ReLU()

            ),  # (256, 8, 8) -> (128, 16, 16)



            'layer3': nn.Sequential(

                spectral_norm(nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1)),

                nn.BatchNorm2d(nch_g),

                nn.ReLU()

            ),  # (128, 16, 16) -> (64, 32, 32)

            'layer4': nn.Sequential(

                spectral_norm(nn.ConvTranspose2d(nch_g, nch, 4, 2, 1)),

                nn.Tanh()

            )   # (64, 32, 32) -> (3, 64, 64)

        })



    def forward(self, z):

        for layer in self.layers.values():  

            z = layer(z)

        return z
class Discriminator(nn.Module):

    def __init__(self, nch=3, nch_d=64):

        super(Discriminator, self).__init__()

        self.layers = nn.ModuleDict({

            'layer0': nn.Sequential(

                spectral_norm(nn.Conv2d(nch, nch_d, 4, 2, 1)),     

                nn.LeakyReLU(negative_slope=0.2)    

            ),  # (3, 64, 64) -> (64, 32, 32)

            'layer1': nn.Sequential(

                spectral_norm(nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1)),

                nn.BatchNorm2d(nch_d * 2),

                nn.LeakyReLU(negative_slope=0.2)

            ),  # (64, 32, 32) -> (128, 16, 16)

            'layer2': nn.Sequential(

                spectral_norm(nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1)),

                nn.BatchNorm2d(nch_d * 4),

                nn.LeakyReLU(negative_slope=0.2)

            ),  # (128, 16, 16) -> (256, 8, 8)

            'layer3': nn.Sequential(

                spectral_norm(nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1)),

                nn.BatchNorm2d(nch_d * 8),

                nn.LeakyReLU(negative_slope=0.2)

            ),  # (256, 8, 8) -> (512, 4, 4)

            'layer4': spectral_norm(nn.Conv2d(nch_d * 8, 1, 4, 1, 0))

            # (512, 4, 4) -> (1, 1, 1)

        })



    def forward(self, x):

        for layer in self.layers.values():  

            x = layer(x)

        return x.squeeze()     

print('device:', device)



netG = Generator(nz=nz, nch_g=nch_g).to(device)

netG.apply(weights_init)    

print(netG)



netD = Discriminator(nch_d=nch_d).to(device)

netD.apply(weights_init)

print(netD)
criterion = nn.MSELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)  



fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)  # save_fake_image用ノイズ（固定）

Loss_D_list, Loss_G_list = [], []



save_fake_image_count = 1
#####    trainig_loop

for epoch in tqdm(range(n_epoch)):

    for itr, data in enumerate(dataloader):

        real_image = data[0].to(device)   # 本物画像

        sample_size = real_image.size(0)  # 画像枚数

        noise = torch.randn(sample_size, nz, 1, 1, device=device)   # 入力ベクトル生成（正規分布ノイズ）       

        real_target = torch.full((sample_size,), 1., device=device)   # 目標値（本物）

        fake_target = torch.full((sample_size,), 0., device=device)   # 目標値（偽物）



        #--------  Update Discriminator  ---------

        netD.zero_grad()    # 勾配の初期化



        output = netD(real_image)   # Discriminatorが行った、本物画像の判定結果

        errD_real = criterion(output, real_target)  # 本物画像の判定結果と目標値（本物）の二乗誤差

        D_x = output.mean().item()  # outputの平均 D_x を計算（後でログ出力に使用）



        fake_image = netG(noise)    # Generatorが生成した偽物画像



        output = netD(fake_image.detach())  # Discriminatorが行った、偽物画像の判定結果

        errD_fake = criterion(output, fake_target)  # 偽物画像の判定結果と目標値（偽物）の二乗誤差

        D_G_z1 = output.mean().item()  # outputの平均 D_G_z1 を計算（後でログ出力に使用）



        errD = errD_real + errD_fake    # Discriminator 全体の損失

        errD.backward()    # 誤差逆伝播

        optimizerD.step()   # Discriminatoeのパラメーター更新



        #---------  Update Generator   ----------

        netG.zero_grad()    # 勾配の初期化        

        output = netD(fake_image)   # 更新した Discriminatorで、偽物画像を判定

        errG = criterion(output, real_target)   # 偽物画像の判定結果と目標値（本物）の二乗誤差

        errG.backward()     # 誤差逆伝播

        D_G_z2 = output.mean().item()  # outputの平均 D_G_z2 を計算（後でログ出力に使用）



        optimizerG.step()   # Generatorのパラメータ更新



        if itr % display_interval == 0:

            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'

                  .format(epoch + 1, n_epoch,itr + 1, len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))



            Loss_D_list.append(errD.item())

            Loss_G_list.append(errG.item())



        if epoch == 0 and itr == 0:     

            vutils.save_image(real_image, '{}/real_samples.png'.format(outf),normalize=True, nrow=8)



        if itr % save_fake_image_interval == 0 and itr > 0:

            fake_image = netG(fixed_noise)

            vutils.save_image(fake_image.detach(), '{}/fake_samples_{:03d}.png'.format(outf, save_fake_image_count),normalize=True, nrow=8)

            save_fake_image_count +=1



    # ---------  save fake image  ----------

    fake_image = netG(fixed_noise)  

    vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),

                      normalize=True, nrow=8)



    # ---------  save model  -----------

    if (epoch + 1) % 10 == 0:   # 10エポックごとにモデルを保存する

        torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))

        torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))



# plot graph

plt.figure()    

plt.plot(range(len(Loss_D_list)), Loss_D_list, color='blue', linestyle='-', label='Loss_D')

plt.plot(range(len(Loss_G_list)), Loss_G_list, color='red', linestyle='-', label='Loss_G')

plt.legend()

plt.xlabel('iter (*100)')

plt.ylabel('loss')

plt.title('Loss_D and Loss_G')

plt.grid()

plt.savefig('Loss_graph.png') 
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

    

im_batch_size = 50

n_images=10000



for i_batch in tqdm(range(0, n_images, im_batch_size)):

    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)

    gen_images = netG(gen_z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))
import shutil

shutil.make_archive('images', 'zip', '../output_images')