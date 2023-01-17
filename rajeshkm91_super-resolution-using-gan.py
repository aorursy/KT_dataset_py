# !conda install numpy pytorch torchvision cpuonly -c pytorch -y

# !pip install matplotlib --upgrade --quiet
# !pip install jovian --upgrade --quiet
import os

import numpy as np

import math

import glob

import random

from time import perf_counter 



import torchvision.transforms as transforms

from torchvision.utils import save_image, make_grid

from torchvision.transforms import ToTensor, ToPILImage

from torchvision.models import vgg19



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.autograd import Variable

from torch.utils.data import Dataset



from PIL import Image
project_name='Super_Resolution_Using_GAN'
# import jovian

# jovian.commit(project=project_name, environment=None)
class FeatureExtractor(nn.Module):

    def __init__(self):

        super(FeatureExtractor, self).__init__()

        

        vgg19_model = vgg19(pretrained=True)

        

        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]) #only using initial 18 layers

        #print(self.feature_extractor)



    def forward(self, img):

        return self.feature_extractor(img)   
class ResidualBlock(nn.Module):

    def __init__(self, in_features):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(in_features, 0.8)

        self.prelu = nn.PReLU()



    def forward(self, x):

        xin = x

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.prelu(x)

        x = self.conv1(x)

        x = self.bn1(x)

        return xin + x
class GeneratorNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):

        super(GeneratorNet, self).__init__()



        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)

        self.prelu = nn.PReLU()

        

        res_blocks = []

        for _ in range(n_residual_blocks):

            res_blocks.append(ResidualBlock(64))

            

        self.res_blocks = nn.Sequential(*res_blocks)



        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(64, 0.8)



        upsampling = []

        for out_features in range(2):

            upsampling += [

                nn.Conv2d(64, 256, 3, 1, 1),

                nn.BatchNorm2d(256),

                nn.PixelShuffle(upscale_factor=2),

                nn.PReLU(),

            ]

        self.upsampling = nn.Sequential(*upsampling)



        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)

        self.tanh = nn.Tanh()



    def forward(self, x):

        out1 = self.prelu(self.conv1(x))

        

        out = self.res_blocks(out1)

        

        out2 = self.bn2(self.conv2(out))

        

        out = torch.add(out1, out2)

        out = self.upsampling(out)

        out = self.tanh(self.conv3(out))

        return out
class Discriminator(nn.Module):

    def __init__(self, input_shape):

        super(Discriminator, self).__init__()



        self.input_shape = input_shape

        in_channels, in_height, in_width = self.input_shape

        

        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)

        self.output_shape = (1, patch_h, patch_w)



        def discriminator_block(in_filters, out_filters, first_block=False):

            layers = []

            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))

            

            if not first_block:

                layers.append(nn.BatchNorm2d(out_filters))

            

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))

            layers.append(nn.BatchNorm2d(out_filters))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            

            return layers



        layers = []

        in_filters = in_channels

        for i, out_filters in enumerate([64, 128, 256, 512]):

            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))

            in_filters = out_filters



        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))



        self.model = nn.Sequential(*layers)



    def forward(self, img):

        return self.model(img)
class opt:

    epoch = 0

    n_epochs = 5

    dataset_name = "img_align_celeba" #https://www.kaggle.com/jessicali9530/celeba-dataset/activity

    batch_size = 4



    lr = 0.0002

    b1 = 0.5

    b2 = 0.999

    n_cpu = 8

    

    hr_height = 256

    hr_width = 256

    channels = 3

    

    sample_interval = 100

    checkpoint_interval = 2

    nrOfImages = 10000
mean = np.array([0.485, 0.456, 0.406])

std = np.array([0.229, 0.224, 0.225])



class ImageDataset(Dataset):

    def __init__(self, root, hr_shape):



        hr_height, hr_width = hr_shape

        

        self.lr_transform = transforms.Compose(

            [

                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),

                transforms.ToTensor(),

                transforms.Normalize(mean, std),

            ]

        )

        self.hr_transform = transforms.Compose(

            [

                transforms.Resize((hr_height, hr_height), Image.BICUBIC),

                transforms.ToTensor(),

                transforms.Normalize(mean, std),

            ]

        )

        #print('root: ', root)

        self.files = sorted(glob.glob(root + "/*.*"))

        #print('self.files: ', self.files)

        self.files = self.files[0:opt.nrOfImages]



    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])

        img_lr = self.lr_transform(img)

        img_hr = self.hr_transform(img)



        return {"lr": img_lr, "hr": img_hr}



    def __len__(self):

        return len(self.files)
os.makedirs("train_images", exist_ok=True)

os.makedirs("saved_models", exist_ok=True)



cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)



generator = GeneratorNet()

discriminator = Discriminator(input_shape = (opt.channels, *hr_shape))

feature_extractor = FeatureExtractor()



feature_extractor.eval()



criterion_GAN = torch.nn.MSELoss()

criterion_content = torch.nn.L1Loss()



if cuda:

    generator = generator.cuda()

    discriminator = discriminator.cuda()

    feature_extractor = feature_extractor.cuda()

    criterion_GAN = criterion_GAN.cuda()

    criterion_content = criterion_content.cuda()



if opt.epoch != 0:

    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch-1))

    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch-1))

    

optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))



Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor



dataloader = DataLoader(

    ImageDataset("../input/celeba-dataset/img_align_celeba/%s" % opt.dataset_name, hr_shape = hr_shape),

    batch_size = opt.batch_size,

    shuffle = True,

    num_workers = opt.n_cpu, )
for epoch in range(opt.epoch, opt.n_epochs):

    for i, imgs in enumerate(dataloader):



        imgs_lr = Variable(imgs["lr"].type(Tensor))

        imgs_hr = Variable(imgs["hr"].type(Tensor))



        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)



        optimizer_G.zero_grad()



        gen_hr = generator(imgs_lr)



        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)



        gen_features = feature_extractor(gen_hr)

        real_features = feature_extractor(imgs_hr)

        loss_content = criterion_content(gen_features, real_features.detach())



        loss_G = loss_content + 1e-3 * loss_GAN



        loss_G.backward()

        optimizer_G.step()



  

        optimizer_D.zero_grad()



        loss_real = criterion_GAN(discriminator(imgs_hr), valid)

        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)



        loss_D = (loss_real + loss_fake) / 2



        loss_D.backward()

        optimizer_D.step()



        print("\n[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item()))



        batches_done = epoch * len(dataloader) + i

        if batches_done % opt.sample_interval == 0:

            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)

            img_grid = torch.cat((imgs_lr, gen_hr), -1)

            save_image(img_grid, "train_images/%d.png" % batches_done, normalize=False)



    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:

        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch) 

        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)



torch.save(generator.state_dict(), "saved_models/generator.pth")

torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")        

    
# jovian.reset()
# jovian.log_hyperparams(start_epoch=opt.epoch,

#                        number_of_epochs=opt.n_epochs,

#                        lrs=opt.lr,

#                        beta1=opt.b1,

#                        beta2=opt.b2)
# jovian.log_metrics(generator_loss=loss_G.item(), discriminator_loss=loss_D.item())
#inference

os.makedirs("images_inference", exist_ok=True)



network = GeneratorNet()

network = network.eval()



if torch.cuda.is_available():

    network.cuda()

    network.load_state_dict(torch.load('saved_models/generator.pth'))

else:

    network.load_state_dict(torch.load('saved_models/generator.pth', map_location=lambda storage, loc: storage))



im_number = '200080'

imgs_lr = Image.open('../input/celeba-dataset/img_align_celeba/img_align_celeba/' + im_number + '.jpg')



imgs_lr = Variable(ToTensor()(imgs_lr)).unsqueeze(0)



if torch.cuda.is_available():

    imgs_lr = imgs_lr.cuda()

    

with torch.no_grad():

    start = perf_counter()

    gen_hr = network(imgs_lr)

    elapsed = (perf_counter() - start)



    print('time cost: ' + str(elapsed) + 'sec')

        

    print('Shape imgs_lr:', imgs_lr.shape)

    print('Shape gen_hr:', gen_hr.shape)

    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

    

    #imgs_lr = ToPILImage()(imgs_lr[0].data.cpu())

    #gen_hr = ToPILImage()(gen_hr[0].data.cpu())

    print('Shape imgs_lr post interpolation:', imgs_lr.shape)



    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)

    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)

    img_grid = torch.cat((imgs_lr, gen_hr), -1)

    save_image(img_grid, "images_inference/"+ str(im_number) + ".png", normalize=False)
# jovian.commit(project=project_name, outputs=["saved_models/generator.pth", "saved_models/discriminator.pth"], environment=None)