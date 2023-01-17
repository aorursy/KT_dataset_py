!nvidia-smi
import os

import random

from PIL import Image

from matplotlib import pyplot as plt



import torch

import torch.nn as nn

import torch.optim as optim



import torchvision

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage
def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])



class DatasetFromfolder(Dataset):

    def __init__(self, path):

        super(DatasetFromfolder, self).__init__()

        self.filenames = []

        self.pathB = '../input/vangogh2photo/trainB/'

        self.foldersB = os.listdir(self.pathB)

        folders = os.listdir(path)

        for f in folders:

            self.filenames.append(path + f)

        self.data_transform = Compose([ToTensor()])



    def __getitem__(self, index):

        imgA = Image.open(self.filenames[index])

        imgA = self.data_transform(imgA)

        

        idx = random.randrange(len(self.foldersB))

        imgB = Image.open(self.pathB + self.foldersB[idx])

        imgB =  self.data_transform(imgB)

        

        return imgA, imgB



    def __len__(self):

        return len(self.filenames)
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        

        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(64)

        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(128)

        self.Conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(256)

        self.Conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.bn5 = nn.BatchNorm2d(512)

        self.Conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

        self.bn6 = nn.BatchNorm2d(1024)

        self.Conv7 = nn.Conv2d(1024, 1, kernel_size=3, stride=2, padding=0)

        

        self.LRU = nn.LeakyReLU(0.2)

        self.sig = nn.Sigmoid()



    def forward(self, x):

        out = self.LRU(self.Conv1(x))

        out = self.LRU(self.bn2(self.Conv2(out)))

        out = self.LRU(self.bn3(self.Conv3(out)))

        out = self.LRU(self.bn4(self.Conv4(out)))

        out = self.LRU(self.bn5(self.Conv5(out)))

        out = self.LRU(self.bn6(self.Conv6(out)))

        out = self.sig(out)        

        

        return out
class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        

        self.Conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)

        self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(64)

        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(128)

        self.Conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(256)

        self.Conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.bn5 = nn.BatchNorm2d(512)        

        

        self.DeConv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)

        self.dbn1 = nn.BatchNorm2d(256)

        self.DeConv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.dbn2 = nn.BatchNorm2d(128)

        self.DeConv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.dbn3 = nn.BatchNorm2d(64)

        self.DeConv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        self.dbn4 = nn.BatchNorm2d(32)

        self.DeConv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        

        self.LRU = nn.LeakyReLU(0.2)

        self.Relu = nn.ReLU()

        self.tanh = nn.Tanh()



    def forward(self, x):

        

        out = self.LRU(self.bn1(self.Conv1(x)))

        out = self.LRU(self.bn2(self.Conv2(out)))

        out = self.LRU(self.bn3(self.Conv3(out)))

        out = self.LRU(self.bn4(self.Conv4(out)))

        out = self.LRU(self.bn5(self.Conv5(out)))

        

        out = self.Relu(self.dbn1(self.DeConv1(out)))    

        out = self.Relu(self.dbn2(self.DeConv2(out)))    

        out = self.Relu(self.dbn3(self.DeConv3(out)))

        out = self.Relu(self.dbn4(self.DeConv4(out)))

        out = self.tanh(self.DeConv5(out))

        

        return out
NUM_EPOCHS = 10001

data_transform = Compose([ToTensor()])

def train():

    

    pathA = '../input/vangogh2photo/trainA/'

    train_set = DatasetFromfolder(pathA)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=32, shuffle=True)

    

    os.system('mkdir checkpoint')

    os.system('mkdir image')



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    G_ab = Generator()

    G_ba = Generator()

    D_a = Discriminator()

    D_b = Discriminator()    

    if torch.cuda.device_count() > 1:

        G_ab = nn.DataParallel(G_ab)

        G_ba = nn.DataParallel(G_ba)

        D_a = nn.DataParallel(D_a)

        D_b = nn.DataParallel(D_b)

    G_ab.to(device)    

    G_ba.to(device)    

    D_a.to(device)    

    D_b.to(device)        

    G_ab_optimizer = optim.Adam(G_ab.parameters())

    G_ba_optimizer = optim.Adam(G_ba.parameters())

    D_a_optimizer = optim.Adam(D_a.parameters())

    D_b_optimizer = optim.Adam(D_b.parameters())

    

    #criterion = nn.L1Loss()

    MSELoss = nn.MSELoss()

    L1Loss = nn.L1Loss()

    

    for epoch in range(NUM_EPOCHS):

        batch_idx = 0

        for A, B in train_loader:

            batch_size = A.size(0)

            

            A = A.to(device)

            B = B.to(device)

            

            G_ab.train()

            G_ba.train()

            D_a.train()

            D_b.train()

            G_ab.zero_grad()

            G_ba.zero_grad()

            D_a.zero_grad()

            D_b.zero_grad()

            

            A_B = G_ab(A)

            B_A = G_ba(B)            

            A_B_A = G_ba(A_B)

            B_A_B = G_ab(B_A)   

            

            real = torch.ones(1).to(device)

            fake = torch.zeros(1).to(device)

            

            D_loss = MSELoss(D_a(A), real) + MSELoss(D_b(B), real) + MSELoss(D_b(A_B), fake) + MSELoss(D_a(B_A), fake)

            D_loss.backward(retain_graph=True)

            D_a_optimizer.step()

            D_b_optimizer.step()

            

            G_loss = L1Loss(A, A_B_A)

            G_loss += L1Loss(B, B_A_B)

            G_loss += MSELoss(D_b(A_B), real) + MSELoss(D_a(B_A), real)

            G_loss.backward(retain_graph=True)

            G_ab_optimizer.step()

            G_ba_optimizer.step()

            

            if batch_idx % 10 == 0:

                print("Epoch:{} batch[{}/{}] G_loss:{} D_loss:{}".format(epoch, batch_idx, len(train_loader), G_loss, D_loss))

                

                img = Image.open('../input/vangogh2photo/testB/2014-08-02 11_46_18.jpg')

                w, h = img.size

                img_ = data_transform(img).unsqueeze(dim=0).to(device)

                img_ab = G_ab(img_)

                img_ba = G_ba(img_ab)

                

                torchvision.utils.save_image(img_, './image/img.png')

                torchvision.utils.save_image(img_ab, './image/AB.png')

                torchvision.utils.save_image(img_ba, './image/BA.png')                

                im1 = Image.open('./image/img.png')

                im2 = Image.open('./image/AB.png')

                im3 = Image.open('./image/BA.png')                 

                dst = Image.new('RGB', (w*3 , h))

                dst.paste(im1, (0, 0))

                dst.paste(im2, (w, 0))

                dst.paste(im3, (w*2, 0))

                plt.imshow(dst)

                plt.title('new Image')

                plt.show()



            batch_idx += 1

    
#train()