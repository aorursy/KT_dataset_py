import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data import DataLoader, Dataset, BatchSampler

import os

import matplotlib.pyplot as plt

from PIL import Image
df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')

df.sample(5)
df.isna().sum()
class eyeData(Dataset):

    def __init__(self,imageDir,csv,transform=None):

        super(eyeData,self).__init__()

        self.dataDirectory = imageDir

        self.tabular = pd.read_csv(csv)

        self.transform = transform

        _, _, files = next(os.walk(self.dataDirectory))

        self.filenames = files

        

    def __len__(self):

        return len(self.filenames)

    

    def __getitem__(self,idx):

        row = self.tabular.iloc[idx].values

        diag = row[1]

        img_name = row[0]

        img_path = os.path.join(self.dataDirectory,img_name+'.png')

        img = Image.open(img_path)

        

        getted = {"image":img,"diagnosis":torch.LongTensor([diag]).unsqueeze(1)}

        if self.transform:

            getted["image"] = self.transform(getted["image"])

        

        return getted

        

EDAdata = eyeData(imageDir='/kaggle/input/aptos2019-blindness-detection/train_images/',

               csv='/kaggle/input/aptos2019-blindness-detection/train.csv')



test = EDAdata[0]
def plotImgwithDiag(obs):

    plt.imshow(obs["image"])

    plt.title(" Diagnosis: {}".format(int(obs["diagnosis"])))

    

plotImgwithDiag(test)
def collateDict(batch):

    collated = {}

    for idx,obs in enumerate(batch):

        collated["BatchIndex_"+str(idx)]=obs

        

    return collated
dataIterator = DataLoader(EDAdata,batch_size=4,shuffle=True,num_workers=4,collate_fn=collateDict)
for j,batch in enumerate(dataIterator):

    plt.figure(figsize=(20,10))

    for idx,I in enumerate(batch.values()):

        plt.subplot(1,4,idx+1)

        plotImgwithDiag(I)

    plt.show()

    if j==2:

        break
df.groupby(["diagnosis"]).count()
from torchvision.models import resnet34



vgg = resnet34(pretrained=False)

vgg.load_state_dict(torch.load('/kaggle/input/resnet34/resnet34.pth'))



for param in vgg.parameters():

    para.requires_grad = False
class VGGEyeClassifier(torch.nn.Module):

    def __init__(self,dropoutRate=0.5):

        super(VGGEyeClassifier,self).__init__()

        self.size = 8

        self.features = torch.nn.Sequential(vgg.conv1,

                                            vgg.bn1,

                                            vgg.relu,

                                            vgg.maxpool,

                                            vgg.layer1,

                                            vgg.layer2,

                                            vgg.layer3,

                                            vgg.layer4,

                                            torch.nn.AdaptiveAvgPool2d(output_size=(self.size,

                                                                                    self.size)))

        self.classifier = torch.nn.Sequential(

            torch.nn.Dropout2d(dropoutRate),

            torch.nn.BatchNorm2d(512),

            torch.nn.Conv2d(

                in_channels=512,

                out_channels=256,

                kernel_size=self.size,

                stride=1,

                padding=0),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(256),

            torch.nn.Dropout2d(dropoutRate),

            torch.nn.Conv2d(

                in_channels=256,

                out_channels=128,

                kernel_size=1,

                stride=1,

                padding=0),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(128),

            torch.nn.Dropout2d(dropoutRate),

            torch.nn.Conv2d(

                in_channels=128,

                out_channels=5,

                kernel_size=1,

                stride=1,

                padding=0))

    

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x
theNet = VGGEyeClassifier(dropoutRate=0.2).cuda()

test = theNet(torch.randn(5,3,256,256).cuda())

print(test.shape)
from torchvision.transforms import functional as F

from torchvision import transforms as T

from torch.utils.data import SubsetRandomSampler

from random import uniform
class RandomBrightness(object):

    def __init__(self,rng=(0.75,1.25)):

        self.rng = rng

    def __call__(self,img):

        rand = uniform(self.rng[0],

                         self.rng[1])

        return F.adjust_brightness(img,rand)

    

class RandomContrast(object):

    def __init__(self,rng=(0.75,1.25)):

        self.rng = rng

    def __call__(self,img):

        rand = uniform(self.rng[0],

                         self.rng[1])

        return F.adjust_contrast(img,rand)



class RandomGamma(object):

    def __init__(self,rng=(0.75,1.25)):

        self.rng = rng

    def __call__(self,img):

        rand = uniform(self.rng[0],

                         self.rng[1])

        return F.adjust_gamma(img,rand)

    

class RandomHue(object):

    def __init__(self,rng=(-0.15,0.15)):

        self.rng = rng

    def __call__(self,img):

        rand = uniform(self.rng[0],

                         self.rng[1])

        return F.adjust_hue(img,rand)

    

class RandomSat(object):

    def __init__(self,rng=(0.75,1.25)):

        self.rng = rng

    def __call__(self,img):

        rand = uniform(self.rng[0],

                         self.rng[1])

        return F.adjust_saturation(img,rand)

    

TrainData = eyeData(imageDir='/kaggle/input/aptos2019-blindness-detection/train_images/',

                    csv='/kaggle/input/aptos2019-blindness-detection/train.csv',

                    transform=T.Compose([T.Resize((256,256)),

                                         T.RandomApply([T.RandomApply([T.RandomAffine(degrees=359,

                                                                        translate=(0.2,0.2),

                                                                        shear=(20,20,20,20)),

                                                        T.RandomHorizontalFlip(1),

                                                        T.RandomVerticalFlip(1)],p=0.7),

                                         T.RandomApply([RandomBrightness(),

                                                        RandomContrast(),

                                                        RandomGamma(),

                                                        RandomSat()],p=0.5)],p=0.7),

                                         T.ToTensor()]))









# data loader will start loading up several batches into memory so my GPU keeps working

dataIteratorTrain = DataLoader(TrainData,

                               batch_size=1,

                               num_workers=4,

                               shuffle=True)



 

print(len(dataIteratorTrain))

for j,batch in enumerate(dataIteratorTrain):

    

    print(batch["image"].shape)

    

    if j>5:

        break
def LookAtImage(image):

    npImage = torch.squeeze(image).detach().cpu().numpy().transpose((1,2,0))

    plt.figure(figsize=(10,10))

    plt.imshow(npImage)

    

for j,batch in enumerate(dataIteratorTrain):

    LookAtImage(batch["image"])

    plt.title("Diagnosis"+str(batch["diagnosis"].detach().numpy()))

    plt.show()

    if j>5:

        break
import time
LR = 1e-3

epochs = 20

theNet = VGGEyeClassifier(0.5).cuda()

dataIteratorTrain = DataLoader(TrainData,

                               batch_size=2,

                               num_workers=2,

                               shuffle=True)

lossFunc = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(filter(lambda x: x.requires_grad,theNet.parameters()),

                         lr=LR)



def passThroughTrain():

    start = time.time()

    print(' ')

    sumLosses = torch.cuda.FloatTensor([0])

    for idx,batch in enumerate(dataIteratorTrain):

        image = batch["image"].cuda()

        diag = batch["diagnosis"].cuda()

        optim.zero_grad()

        pred = theNet(image)

        loss = lossFunc(pred,diag)

        loss.backward()

        optim.step()

        sumLosses += loss.detach()

        print("Batch {}/{}".

              format((idx+1),(len(dataIteratorTrain))),end='\r')

    print("Mean Training Loss: {}".format(sumLosses.detach().cpu().numpy()/len(dataIteratorTrain)))

    print("Epoch time: {}".format(time.time()-start))

    return sumLosses.detach().cpu().numpy()/len(dataIteratorTrain)





TrainingLosses = []

print("Beginning training for {} epochs".format(epochs))

for i in range(epochs):

    print("Epoch"+str(i+1))

    TrainingLosses.append(passThroughTrain())
plt.plot(TrainingLosses)