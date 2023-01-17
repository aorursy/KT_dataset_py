import os

import tarfile

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

from IPython.display import FileLink



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset



import torchvision

from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize
tar = tarfile.open('../input/cifar.tgz')

# tar.extractall()

tar.close()

cifar_path = './cifar/train'
def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])



class DatasetFromfolder(Dataset):

    def __init__(self, path):

        super(DatasetFromfolder, self).__init__()

        path = './cifar/train/'

        self.filenames = []

        label = 0

        folders = os.listdir(path)

        for f in folders:

            self.filenames.append([path + f, label])

        self.data_transform = Compose([ToTensor()])



    def __getitem__(self, index):

        resultimage = self.data_transform(Image.open(self.filenames[index][0]))

        label = torch.ones(1) * self.filenames[index][1]

        return resultimage, label



    def __len__(self):

        return len(self.filenames)
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.Conv1 = nn.Conv2d(3, 128, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(128)

        self.Conv2 = nn.Conv2d(128, 256, 4, 2, 0)

        self.bn2 = nn.BatchNorm2d(256)

        self.Conv3 = nn.Conv2d(256, 512, 4, 1, 0)

        self.bn3 = nn.BatchNorm2d(512)

        self.Conv4 = nn.Conv2d(512, 1, 4, 1, 0)

        self.LRU = nn.LeakyReLU(0.2)

        self.sig = nn.Sigmoid()



    def forward(self, z):

        z = self.LRU(self.bn1(self.Conv1(z)))

        z = self.LRU(self.bn2(self.Conv2(z)))

        z = self.LRU(self.bn3(self.Conv3(z)))

        out = self.sig(self.LRU(self.Conv4(z)))

        return out



class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()



        self.Conv1 = nn.Conv2d(1, 128, 3, 2, 1)

        self.bn1 = nn.BatchNorm2d(128)

        self.Conv2 = nn.Conv2d(128, 128, 3, 2, 1)

        self.bn2 = nn.BatchNorm2d(128)

        self.Conv3 = nn.Conv2d(128, 128, 3, 2, 1)

        self.bn3 = nn.BatchNorm2d(128)

        self.Conv4 = nn.Conv2d(128, 128, 3, 2, 1)

        self.bn4 = nn.BatchNorm2d(128)

        self.Conv5 = nn.Conv2d(128, 128, 3, 2, 1)

        self.bn5 = nn.BatchNorm2d(128)



        self.DeConv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.dbn1 = nn.BatchNorm2d(128)

        self.DeConv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.dbn2 = nn.BatchNorm2d(128)

        self.DeConv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.dbn3 = nn.BatchNorm2d(128)

        self.DeConv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.DeConv5 = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)



        self.LRU = nn.LeakyReLU(0.2)

        self.Relu = nn.ReLU()

        self.tanh = nn.Tanh()



    def forward(self, z):

        _out1 = self.bn1(self.Conv1(z))

        out1 = self.LRU(_out1)

        _out2 = self.bn2(self.Conv2(out1))

        out2 = self.LRU(_out2)

        _out3 = self.bn3(self.Conv3(out2))

        out3 = self.LRU(_out3)

        _out4 = self.bn4(self.Conv4(out3))

        out4 = self.LRU(_out4)

        _out5 = self.bn5(self.Conv5(out4))

        out5 = self.LRU(_out5)



        out6 = self.Relu(self.dbn1(self.DeConv1(out5)))

        out6 = torch.cat([_out4, out6], 1)

        out7 = self.Relu(self.dbn2(self.DeConv2(out6)))

        out7 = torch.cat([_out3, out7], 1)

        out8 = self.Relu(self.dbn3(self.DeConv3(out7)))

        out8 = torch.cat([_out2, out8], 1)

        out9 = self.Relu(self.DeConv4(out8))

        out9 = torch.cat([_out1, out9], 1)

        out10 = self.Relu(self.DeConv5(out9))



        out = self.tanh(out10)



        return out
def train():

    NUM_EPOCHS = 1 #10

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_set = DatasetFromfolder('./cifar/train/')

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=64, shuffle=True)



    netG = Generator()

    netD = Discriminator()

    if torch.cuda.device_count() > 1:

        netG = nn.DataParallel(netG)

        netD = nn.DataParallel(netD)

    netG.to(device)

    netD.to(device)



    optimizerG = optim.Adam(netG.parameters())

    optimizerD = optim.Adam(netD.parameters())

    criterion = nn.MSELoss().to(device)

    criterion2 = nn.L1Loss().to(device)

    

    os.system('mkdir checkpoint')

    os.system('mkdir image')

    new_point = 0



    for epoch in range(1, NUM_EPOCHS + 1):

        batch_idx = 0

        for x, label in train_loader:

            x = 2*x - 1

            x = x.to(device)

            #label.to(device)

            batch_size = x.size(0)



            chan_r = x[:, 0, :, :]

            chan_g = x[:, 1, :, :]

            chan_b = x[:, 2, :, :]

            x_gray = ((chan_r + chan_g + chan_b) / 3).unsqueeze(dim=1)



            fake_image = netG(x_gray)



            real = netD(x).to(device)

            fake = netD(fake_image).to(device)

            real_label = torch.ones(batch_size).to(device)

            fake_label = torch.zeros(batch_size).to(device)



            netD.train()

            netD.zero_grad()

            d_loss = criterion(fake.squeeze(), fake_label) + criterion(real.squeeze(), real_label)

            d_loss.backward(retain_graph=True)

            optimizerD.step()



            netG.train()

            netG.zero_grad()

            g_loss = criterion2(x, fake_image) + (d_loss * 10)

            g_loss.backward(retain_graph=True)

            optimizerG.step()



            if batch_idx % 100 == 0:

                netG.eval()

                generated_image = netG(x_gray)

                generated_image = (generated_image + 1) / 2



                print("Epoch:{} batch[{}/{}] G_loss:{} D_loss:{}".format(epoch, batch_idx, len(train_loader), g_loss, d_loss))

                torchvision.utils.save_image(generated_image.data, './image/Generated.png')

                

                plt.imshow(Image.open('./image/Generated.png'))

                plt.title('Generated Image')

                plt.show()

                

            batch_idx += 1

                

        torch.save(netG.state_dict(), './checkpoint/netGckpt_%d.pth' % (new_point))

        torch.save(netD.state_dict(), './checkpoint/netDckpt_%d.pth' % (new_point))

        new_point += 1
#train()