import torch

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt


class tinyUNet(torch.nn.Module):

    def __init__(self,in_channels=3,

                 filters1=64,filters2 = 128,filters3=256,

                 out_classes=5,

                 filter_size=3,Pools=4):

        super(tinyUNet,self).__init__()

        self.Conv1 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(in_channels),

            torch.nn.Conv2d(

                in_channels=in_channels,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Down1 = torch.nn.MaxPool2d(Pools)

        self.Conv2 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters1),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Down2 = torch.nn.MaxPool2d(Pools)

        self.Conv3 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters2),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters3,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters3,

                out_channels=filters3,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Up1 = torch.nn.ConvTranspose2d(

            in_channels=filters3,

            out_channels=filters2,

            kernel_size=Pools,

            stride=Pools,

            padding=0)

        self.Conv4 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters2+filters2),

            torch.nn.Conv2d(

                in_channels=filters2+filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Up2 = torch.nn.ConvTranspose2d(

            in_channels=filters2,

            out_channels=filters1,

            kernel_size=Pools,

            stride=Pools,

            padding=0)

        self.Conv5 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters1+filters1),

            torch.nn.Conv2d(

                in_channels=filters1+filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Out = torch.nn.Conv2d(in_channels=filters1,

                                   out_channels=out_classes,

                                   kernel_size=1,

                                   padding=0,

                                   stride=1)

    

    def forward(self,Input):

        self.Conv1Out = self.Conv1((Input))

        self.Down1Out = self.Down1(self.Conv1Out)

        self.Conv2Out = self.Conv2(self.Down1Out)

        self.Down2Out = self.Down2(self.Conv2Out)

        self.Conv3Out = self.Conv3(self.Down2Out)

        self.Up1Out = self.Up1(self.Conv3Out)

        self.Conv4Out = self.Conv4(torch.cat((self.Conv2Out,self.Up1Out),dim=1))

        self.Up2Out = self.Up2(self.Conv4Out)

        self.Conv5Out = self.Conv5(torch.cat((self.Conv1Out,self.Up2Out),dim=1))

        self.Logit = self.Out(self.Conv5Out)

        return self.Logit

  
Model = tinyUNet().to('cuda')
Model.load_state_dict(torch.load('/kaggle/input/sss-v2-model-training-data-augmentation-no-reg/miniUNet8'))
Model.eval()

Model
! ls /kaggle/input/severstal-steel-defect-detection/ -al
# Thanks @rakhlin for sharing!

# https://www.kaggle.com/rakhlin/fast-run-length-encoding-python



def rle_encoding(x):

    '''

    x: numpy array of shape (height, width), 1 - mask, 0 - background

    Returns run length as list

    '''

    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths
!mkdir /kaggle/test_masks
for dirname,_,filenames in os.walk('/kaggle/input/severstal-steel-defect-detection/test_images/'):

    files = len(filenames)

    for i in range(0,files):

        Istack = []

        Istack.append(plt.imread(dirname+filenames[i]))

        testTensor = torch.cuda.FloatTensor(

            np.swapaxes(

                np.swapaxes(

                    np.stack(

                        Istack,axis=0),1,3),2,3))

        pred = Model(testTensor).detach().cpu().numpy()

        np.save('/kaggle/test_masks/'+str(filenames[i]),pred)

        print("Completion: {}%".format(100*(i+1)/files),end='\r')
Predictions = []



for dirname,_,filenames in os.walk('/kaggle/test_masks/'):

    for idx,filename in enumerate(filenames):

        Masks = np.load(dirname+filename)

        MasksHWC = np.squeeze(np.argmax(Masks,axis=1))

        for i in range(4):

            RLE = str(rle_encoding(MasksHWC==i)).replace(',','').replace('[','').replace(']','')

            Predictions.append([filename[:-4]+'_'+str(i+1),RLE])

        print("Completion : {}%".format(100*(idx+1)/len(filenames)),end='\r')

PDF = pd.DataFrame(Predictions,columns = ['ImageId_ClassId','EncodedPixels'])
PDF.head()
PDF.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv').head(40)