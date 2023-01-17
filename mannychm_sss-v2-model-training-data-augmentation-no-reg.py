import numpy as np

import torch

import matplotlib.pyplot as plt

import pandas as pd

import os

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

from torch.utils.data.sampler import SubsetRandomSampler
def rleToMask(rleString,height,width):

    rows,cols = height,width

    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]

    rlePairs = np.array(rleNumbers).reshape(-1,2)

    img = np.zeros(rows*cols,dtype=np.uint8)

    for index,length in rlePairs:

        index -= 1

        img[index:index+length] = 1

    img = img.reshape(cols,rows)

    img = img.T

    return img
DF = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')

DF.head()
def showSegmentation(image,rle):

    mask = rleToMask(rle,image.shape[0],image.shape[1])

    plt.figure()

    plt.imshow(image)

    plt.figure()

    plt.imshow(mask)

    
n = 5

img_name = DF[~DF.EncodedPixels.isna()].iloc[n,0][:-2]

rlePixels = DF[~DF.EncodedPixels.isna()].iloc[n,1]



showSegmentation(plt.imread('/kaggle/input/severstal-steel-defect-detection/train_images/'+img_name),rlePixels)
def LabelsToNumpyMasks(Labels,height,width):

    Tensor = np.zeros((height,width,len(Labels)),dtype=np.uint8)

    for name,rle in Labels.values:

        if str(rle) != 'nan':

#             print(rle)

            Tensor[:,:,int(name[-1])-1]=rleToMask(rle,height,width)

        else:

            pass

    NoClass = (Tensor.sum(axis=2,keepdims=1)<1).astype(np.uint8)

    TotalTensor = np.dstack((Tensor,NoClass))

    return TotalTensor
class SteelDefectSegmentationDataset(Dataset):



    def __init__(self, csv_file, root_dir, transform=None):

        

        self.segmentationRLE = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

        _, _, files = next(os.walk(self.root_dir))

        self.filenames = files



    def __len__(self):

        

        return len(self.filenames)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,

                                self.filenames[idx])

        image = plt.imread(img_name)

        Labels = self.segmentationRLE[self.segmentationRLE.ImageId_ClassId.str.contains(self.filenames[idx])]

        masks = LabelsToNumpyMasks(Labels,image.shape[0],image.shape[1])

        sample = {'name':img_name,'image': image, 'masks': masks}



        if self.transform:

            sample = self.transform(sample)



        return sample


class ToTensor(object):

    def __call__(self, sample):

        image, masks = sample['image'], sample['masks']

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        masks = masks.transpose((2,0,1))

        return {'name':sample['name'],

                'image': torch.FloatTensor(image),

                'masks': torch.argmax(torch.FloatTensor(masks),dim=0)}


steel_dataset = SteelDefectSegmentationDataset(csv_file='/kaggle/input/severstal-steel-defect-detection/train.csv',

                                               root_dir='/kaggle/input/severstal-steel-defect-detection/train_images/',transform=ToTensor())
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

        self.Conv1Out = self.Conv1(Input)

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
class RandomRotateTensor(object):

    def __call__(self,sample):

        image,masks = sample['image'],sample['masks']

        

        randomRotation = np.random.randint(0,6)

        if randomRotation == 0:

            return {'name':sample['name'],

                    'image':image.flip(2).transpose(3,2),

                    'masks':masks.flip(1).transpose(2,1)}

        if randomRotation == 1:

            return {'name':sample['name'],

                    'image':image.transpose(3,2).flip(2),

                    'masks':masks.transpose(2,1).flip(1)}

        if randomRotation == 2:

            return {'name':sample['name'],

                    'image':image.flip(3),

                    'masks':masks.flip(2)}

        if randomRotation == 3:

            return {'name':sample['name'],

                    'image':image.flip(2),

                    'masks':masks.flip(1)}

        if randomRotation == 4:

            return {'name':sample['name'],

                    'image':image.flip(3).flip(2),

                    'masks':masks.flip(2).flip(1)}

        if randomRotation == 5:

            return {'name':sample['name'],

                    'image':image,

                    'masks':masks}

        

Rotator = RandomRotateTensor()

batch_size = 4

validation_split = .1

# Creating data indices for training and validation splits:

dataset_size = len(steel_dataset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

np.random.shuffle(indices)



train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(steel_dataset, batch_size=batch_size, 

                                           sampler=train_sampler,num_workers=batch_size)

validation_loader = torch.utils.data.DataLoader(steel_dataset, batch_size=batch_size,

                                                sampler=valid_sampler,num_workers=batch_size)

theNet = tinyUNet().to('cuda')

torch.cuda.empty_cache()

LR = 5e-3

lossFunc = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(theNet.parameters(),lr=LR)

epochs = 10

ValidationLosses = []

count=1

for i in range(epochs):

    print(' ')

    for idx,batch in enumerate(train_loader):

        batch['image'] = batch['image'].cuda()

        batch['masks'] = batch['masks'].cuda()

        rotBatch = Rotator(batch)

        pred = theNet(rotBatch['image'])

        loss = lossFunc(pred,rotBatch['masks'])

        loss.backward()

        optim.step()

        optim.zero_grad()



        print(' Training Completion: {}%'.

              format(round((count)/(epochs*len(train_loader))*100,3)),end='\r')

        count+=1

        del pred, loss

    print(' ')

    sumLosses=0

    with torch.set_grad_enabled(False):

        for j,batch in enumerate(validation_loader):

            pred = theNet(batch['image'].to('cuda'))

            loss = lossFunc(pred,batch['masks'].to('cuda'))

            sumLosses+=loss.detach().cpu().numpy()

            del pred, loss

            print(' Validating: {}%'.format(round((j+1)/len(validation_loader)*100)),end='\r')

    ValidationLoss=sumLosses/len(validation_loader)

    print(' Validation Loss: {}'.format(round(ValidationLoss,5)))

    ValidationLosses.append(ValidationLoss)

    torch.save(theNet.state_dict(),'/kaggle/working/miniUNet'+str(i))



plt.figure()

plt.plot(ValidationLosses)

plt.show()



print('Validation Loss was lowest for epoch index: {}'.

      format(np.argmin(np.array(ValidationLosses))))
! ls 