import os

from PIL import Image

from matplotlib import pyplot as plt

import torch

from torch import optim

from torch import nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import torchvision

from torchvision import transforms

from torchvision.transforms import Compose, ToTensor
def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])



class DatasetFromfolder(Dataset):

    def __init__(self, path):

        super(DatasetFromfolder, self).__init__()

        folders = os.listdir(path)

        self.filenames = []

        for folder in folders:

            label = int(folder)

            folder = os.path.join(path, folder)

            for x in os.listdir(folder):

                self.filenames.append([os.path.join(folder, x), label])

        self.data_transform = Compose([ToTensor()])



    def __getitem__(self, index):

        resultimage = self.data_transform(Image.open(self.filenames[index][0]).convert('L'))

        label = torch.ones(1) * self.filenames[index][1]

        return resultimage, label



    def __len__(self):

        return len(self.filenames)
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.Conv1 = nn.Conv2d(11, 128, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(128)

        self.Conv2 = nn.Conv2d(128, 256, 4, 2, 1)

        self.bn2 = nn.BatchNorm2d(256)

        self.Conv3 = nn.Conv2d(256, 512, 4, 1, 0)

        self.bn3 = nn.BatchNorm2d(512)

        self.Conv4 = nn.Conv2d(512, 1, 4, 1, 0)

        self.LRU = nn.LeakyReLU(0.2)

        self.sig = nn.Sigmoid()



    def forward(self, z, label):

        z = torch.cat([z, label], 1)

        z = self.LRU(self.bn1(self.Conv1(z)))

        z = self.LRU(self.bn2(self.Conv2(z)))

        z = self.LRU(self.bn3(self.Conv3(z)))

        out = self.sig(self.LRU(self.Conv4(z)))



        return out



class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.DeConv1 = nn.ConvTranspose2d(110, 512, kernel_size=4, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(512)

        self.DeConv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(256)

        self.DeConv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(128)

        self.DeConv4 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)

        self.Relu = nn.ReLU()

        self.tanh = nn.Tanh()



    def forward(self, z, label):

        z = torch.cat([z, label], 1)

        out1 = self.Relu(self.bn1(self.DeConv1(z)))

        out2 = self.Relu(self.bn2(self.DeConv2(out1)))

        out3 = self.Relu(self.bn3(self.DeConv3(out2)))

        out4 = self.tanh(self.DeConv4(out3))



        return out4

NUM_EPOCHS = 100

def train():

    path = '../input/mnist/mnist/training'

    train_set = DatasetFromfolder(path)

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=16, shuffle=True)

    

    os.system('mkdir checkpoint')

    os.system('mkdir image')



    netG = Generator()

    netD = Discriminator()

    optimizerG = optim.Adam(netG.parameters())

    optimizerD = optim.Adam(netD.parameters())

    criterion = nn.MSELoss()

    

    onehot = torch.zeros(10, 10)

    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)

    fill = torch.zeros([10, 10, 28, 28])

    for i in range(10):

        fill[i, i, :, :] = 1



    for epoch in range(NUM_EPOCHS):

        batch_idx = 0

        for real_image, label in train_loader:



            batch_size = real_image.size(0)



            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)

            y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()

            y_label_ = onehot[y_]

            y_fill_ = fill[y_]



            fake_image = netG(z_, y_label_)

            fake_result = netD(fake_image, y_fill_)



            label = label.squeeze()

            label = label.type(torch.LongTensor)

            y_label_ = onehot[label]

            y_fill_ = fill[label]



            real_result = netD(real_image, y_fill_)

            netD.train()

            netD.zero_grad()

            d_loss = criterion(fake_result.squeeze(), torch.zeros(batch_size)) + criterion(real_result.squeeze(), torch.ones(batch_size))

            d_loss.backward(retain_graph=True)

            optimizerD.step()



            netG.train()

            netG.zero_grad()

            g_loss = criterion(fake_result.squeeze(), torch.ones(batch_size))

            g_loss.backward(retain_graph=True)

            optimizerG.step()



            if batch_idx % 20 == 0:

                netG.eval()

                eval_z = torch.rand(batch_size, 100, 1, 1)

                generated_image = netG(eval_z, y_label_)

                generated_image = (generated_image + 1) / 2

                print("Epoch:{} batch[{}/{}] G_loss:{} D_loss:{}".format(epoch, batch_idx, len(train_loader), g_loss, d_loss))

                torchvision.utils.save_image(generated_image.data, './image/Generated.png')

                plt.imshow(Image.open('./image/Generated.png'))

                plt.title('new Image')

                plt.show()



            batch_idx += 1

#train()