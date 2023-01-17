import os

import tarfile

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

        self.data_transform = Compose([RandomCrop([33, 33]), ToTensor()])

        self.data_transform_PIL = Compose([ToPILImage()])



    def __getitem__(self, index):        

        w = h = 33

        img = Image.open(self.filenames[index])

        img, _cb, _cr = img.convert('YCbCr').split()     

        img = self.data_transform(img)           

        result_image = img

        

        resize_image = self.data_transform_PIL(img)        

        resize_image = resize_image.resize((int(w/3), int(h/3)))

        resize_image = resize_image.resize((w, h), Image.BICUBIC)

        resize_image = self.data_transform(resize_image) 

        

        return result_image, resize_image



    def __len__(self):

        return len(self.filenames)
class SuperResolutionCNN(nn.Module):

    def __init__(self):

        super(SuperResolutionCNN, self).__init__()

        self.Conv1 = nn.Conv2d(1, 64, 9, 1, 4)

        self.Conv2 = nn.Conv2d(64, 32, 3, 1, 1)

        self.Conv3 = nn.Conv2d(32, 1, 5, 1, 2)

        self.Relu = nn.ReLU()

        self.sig = nn.Sigmoid()



    def forward(self, x):

        out = self.Relu(self.Conv1(x))

        out = self.Relu(self.Conv2(out))

        out = self.Conv3(out)

        return out
def train():

    NUM_EPOCHS = 1 #501

    data_transform = Compose([ToTensor()])    

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_set = DatasetFromfolder('../input/super-resolution-dataset/Train/')

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=32, shuffle=True)

    

    SRCNN = SuperResolutionCNN()

    if torch.cuda.device_count() > 1:

        SRCNN = nn.DataParallel(SRCNN)

    SRCNN.to(device)    

    

    optimizer = optim.Adam(SRCNN.parameters())

    criterion = nn.MSELoss().to(device)

    

    new_point = 0

    os.system('mkdir checkpoint')

    os.system('mkdir image')

    

    for epoch in range(NUM_EPOCHS):        

        batch_idx = 0        

        for HR, LR in train_loader:

            HR = HR.to(device)

            LR = LR.to(device)            

            newHR = SRCNN(LR) 

            

            SRCNN.train()

            SRCNN.zero_grad()

            loss = criterion(HR, newHR)

            loss.backward(retain_graph=True)

            optimizer.step()

            

            if epoch%50==0 and batch_idx%1==0:

                SRCNN.eval()

                print("Epoch:{} batch[{}/{}] loss:{}".format(epoch, batch_idx, len(train_loader), loss))      

                

                img = Image.open('../input/super-resolution-dataset/Test/butterfly_GT.bmp')   

                            

                w, h = img.size

                

                result_image = img

                result_image_y, _cb, _cr = result_image.convert('YCbCr').split()

                result_image_y = data_transform(result_image_y)   

        

                resize_image = img.resize((int(w/3), int(h/3)), Image.BICUBIC)       

                resize_image = resize_image.resize((w, h), Image.BICUBIC)

                resize_image_y, _cb, _cr = resize_image.convert('YCbCr').split()

                resize_image_y = data_transform(resize_image_y).to(device)

                newHR = SRCNN(resize_image_y.unsqueeze(0))

                

                torchvision.utils.save_image(resize_image_y, './image/LR.png')

                torchvision.utils.save_image(result_image_y, './image/HR.png')

                torchvision.utils.save_image(newHR, './image/newHR.png')

                

                im1 = Image.open('./image/LR.png')

                im2 = Image.open('./image/HR.png')

                im3 = Image.open('./image/newHR.png')                

                dst = Image.new('RGB', (w*3 , h))

                dst.paste(im1, (0, 0))

                dst.paste(im2, (w, 0))

                dst.paste(im3, (w*2, 0))

                dst.save('./image/image.png')

                img = Image.open('./image/image.png')

                plt.imshow(img)

                plt.title('new Image')

                plt.show()

                

            batch_idx += 1

            

        torch.save(SRCNN.state_dict(), './checkpoint/ckpt_%d.pth' % (new_point))

        new_point += 1
train()
Image.open('./image/image.png')
from IPython.display import FileLink

#FileLink('./image/image.png')

#FileLink('./image/LR.png')

#FileLink('./image/HR.png')

#FileLink('./image/newHR.png')