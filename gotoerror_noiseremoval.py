import os

import numpy as np

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

        folders = os.listdir(path)

        for f in folders:

            self.filenames.append(path + f)

        self.data_transform = Compose([ToTensor()])

        self.data_transform_PIL = Compose([RandomCrop([33, 33])])



    def __getitem__(self, index):   

        img = Image.open(self.filenames[index])

        img = self.data_transform_PIL(img)

        

        clear_image = self.data_transform(img)

        

        temp = './jpeg/temp.jpg'        

        img.save(temp, format='JPEG', subsampling=0, quality=10)

        noise_image = Image.open(temp)

        noise_image = self.data_transform(noise_image)

                                           

        return clear_image, noise_image



    def __len__(self):

        return len(self.filenames)
class NoiseRemovalCNN(nn.Module):

    def __init__(self):

        super(NoiseRemovalCNN, self).__init__()

        self.Conv1 = nn.Conv2d(3, 64, 9, 1, 4)

        self.Conv2 = nn.Conv2d(64, 32, 7, 1, 3)

        self.Conv3 = nn.Conv2d(32, 3, 1, 1, 0)

        self.Relu = nn.ReLU()

        self.sig = nn.Sigmoid()



    def forward(self, x):

        out = self.Relu(self.Conv1(x))

        out = self.Relu(self.Conv2(out))

        out = self.Conv3(out)

        return out
def train():

    NUM_EPOCHS = 1 #10001

    data_transform_PIL = Compose([ToPILImage()])

    data_transform = Compose([ToTensor()])

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_set = DatasetFromfolder('../input/super-resolution-dataset/Train/')

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=64, shuffle=True)

    

    model = NoiseRemovalCNN()

    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)

    model.to(device)    

    

    optimizer = optim.Adam(model.parameters())

    criterion = nn.MSELoss().to(device)

    

    new_point = 0

    os.system('mkdir checkpoint')

    os.system('mkdir image')

    os.system('mkdir jpeg')

    

    for epoch in range(NUM_EPOCHS):        

        batch_idx = 0        

        for clear_image, noise_image in train_loader:

            

            clear_image = clear_image.to(device)

            noise_image = noise_image.to(device)            

            new_image = model(noise_image) 

            

            model.train()

            model.zero_grad()

            loss = criterion(clear_image, new_image)

            loss.backward(retain_graph=True)

            optimizer.step()

            

            if batch_idx % 50 == 0:

                model.eval()

                print("Epoch:{} batch[{}/{}] loss:{}".format(epoch, batch_idx, len(train_loader), loss))                

                

                img = Image.open('../input/super-resolution-dataset/Test/ppt3.bmp') 

                w, h = img.size

                

                clear_image = data_transform(img)

                

                temp = './jpeg/test.jpg'

                img.save(temp, format='JPEG', subsampling=0, quality=10)

                noise_image = Image.open(temp)

                noise_image = data_transform(noise_image)  

                noise_image = noise_image.to(device)

                

                new_image = model(noise_image.unsqueeze(dim=0))              

                

                torchvision.utils.save_image(noise_image, './image/noise.png')

                torchvision.utils.save_image(new_image, './image/new.png')

                torchvision.utils.save_image(clear_image, './image/clear.png')

                

                im1 = Image.open('./image/noise.png')

                im2 = Image.open('./image/new.png')

                im3 = Image.open('./image/clear.png')                

                dst = Image.new('RGB', (w*3 , h))

                dst.paste(im1, (0, 0))

                dst.paste(im2, (w, 0))

                dst.paste(im3, (w*2, 0))

                dst.save('./image/image.png')

                img = Image.open('./image/image.png')

                plt.title('new Image')

                plt.imshow(img)

                plt.show()

                

            batch_idx += 1

            

        torch.save(model.state_dict(), './checkpoint/ckpt_%d.pth' % (new_point))

        new_point += 1
train()