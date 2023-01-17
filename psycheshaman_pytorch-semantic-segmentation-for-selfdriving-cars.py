import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt
np.unique(plt.imread('../input/lyft-udacity-challenge/dataA/dataA/CameraSeg/02_00_000.png')[:,:,0]*255)
plt.figure()



plt.imshow(plt.imread('../input/lyft-udacity-challenge/dataA/dataA/CameraRGB/02_00_000.png'))

plt.title('RGB image')

plt.show()

#subplot(r,c) provide the no. of rows and columns

#f, axarr = plt.subplots(1,13,figsize=(15,12)) 



# use the created array to output your multiple images. In this case I have stacked 4 images vertically



labels = ['Unlabeled','Building','Fence','Other',

                 'Pedestrian', 'Pole', 'Roadline', 'Road',

                 'Sidewalk', 'Vegetation', 'Car','Wall',

                 'Traffic sign']



for i in range(13):

    mask = plt.imread('../input/lyft-udacity-challenge/dataA/dataA/CameraSeg/02_00_000.png')*255

    mask = np.where(mask == i, 255, 0)

    mask = mask[:,:,0]

    #axarr[i].imshow(mask)

    plt.title(labels[i])

    plt.imshow(mask)

    plt.show()
cameraRGB = []

cameraSeg = []

for root, dirs, files in os.walk('../input/lyft-udacity-challenge'):

    for name in files:

        f = os.path.join(root, name)

        if 'CameraRGB' in f:

            cameraRGB.append(f)

        elif 'CameraSeg' in f:

            cameraSeg.append(f)

        else:

            break
df = pd.DataFrame({'cameraRGB': cameraRGB, 'cameraSeg': cameraSeg})

df.sort_values(by='cameraRGB',inplace=True)

df.reset_index(drop=True, inplace=True)

df.head(5)
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torch
from torch.nn import functional as F
np.unique(F.interpolate(input=torch.from_numpy(

                               plt.imread('../input/lyft-udacity-challenge/dataA/dataA/CameraSeg/02_00_000.png')[:,:,0]*255).\

                        unsqueeze(0),

             size=256,

             mode='nearest'))
class CustomDatasetFromImages(Dataset):

    def __init__(self, data_info):

        self.data_info = data_info

        self.image_arr = self.data_info.iloc[:, 0]

        self.label_arr = self.data_info.iloc[:, 1]

        self.data_len = len(self.data_info.index)



    def __getitem__(self, index):

        img = np.asarray(Image.open(self.image_arr[index])).astype('float')

        img = torch.as_tensor(img)/255.

        img = img.unsqueeze(0).permute(0,3,1,2)

        img = F.interpolate(input=img,size=256,align_corners=False,mode='bicubic')

        

        lab = np.asarray(plt.imread(self.label_arr[index])).astype('float')[:,:,0]*255

        lab = torch.as_tensor(lab).unsqueeze(0)

        lab = lab.unsqueeze(0)#.permute(0,3,1,2)

        lab = F.interpolate(input=lab,size=256,mode='nearest')



        return (img.float(), lab.float())



    def __len__(self):

        return self.data_len
from sklearn.model_selection import train_test_split



X_train, X_test = train_test_split(df,test_size=0.3)



X_train.reset_index(drop=True,inplace=True)

X_test.reset_index(drop=True,inplace=True)
train_data = CustomDatasetFromImages(X_train)

test_data = CustomDatasetFromImages(X_test)
train_data_loader = DataLoader(train_data,batch_size=1,shuffle=True)

test_data_loader = DataLoader(test_data,batch_size=1,shuffle=False)
!pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp
model = smp.Unet('resnet34', classes=13, activation='softmax')
model = model.float()
def dice_loss(output, target, weights=None, ignore_index=None):

    """

    output : NxCxHxW Variable

    target :  NxHxW LongTensor

    weights : C FloatTensor

    ignore_index : int index to ignore from loss

    """

    eps = 0.0001



    output = output.float().exp()

    target = target.type(torch.int64)

    encoded_target = output.detach() * 0

    if ignore_index is not None:

        mask = target == ignore_index

        target = target.clone()

        target[mask] = 0

        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        mask = mask.unsqueeze(1).expand_as(encoded_target)

        encoded_target[mask] = 0

    else:

        encoded_target.scatter_(1, target.unsqueeze(1), 1)



    if weights is None:

        weights = 1



    intersection = output * encoded_target

    numerator = 2 * intersection.sum(0).sum(1).sum(1)

    denominator = output + encoded_target



    if ignore_index is not None:

        denominator[mask] = 0

    denominator = denominator.sum(0).sum(1).sum(1) + eps

    loss_per_channel = weights * (1 - (numerator / denominator))



    return loss_per_channel.sum() / output.size(1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model.cuda()
for epoch in range(5):  # loop over the dataset multiple times



    running_loss = 0.0

    epoch_loss = []

    t = 0

    for i, data in enumerate(train_data_loader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data

        inputs = inputs.cuda()

        labels = labels.cuda()





        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = model(inputs[0])

        #print(outputs.shape)

        #print(labels[0,0,0,:,:].shape)

        loss = dice_loss(outputs,labels[0,0,:,:,:])

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        epoch_loss.append(loss.item())

        t+=1

        if t % 5000 == 4999:    # print every 5000 mini-batches



            samp = int(np.random.randint(low=0,high=len(test_data)-1,size=1))

            test = model(test_data[samp][0].cuda())

            truth = test_data[samp][1].cuda()

            

            plt.figure()



            #subplot(r,c) provide the no. of rows and columns

            f, axarr = plt.subplots(1,2) 

            plt.suptitle(f'Epoch: {epoch}, batchcount: {t}, avg. loss for last 5000 images: {running_loss/5000}')

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically

            axarr[0]. imshow(torch.argmax(test.squeeze(), dim=0).detach().cpu().numpy())

            axarr[0].set_title('Guessed labels')

            

            

            axarr[1].imshow(truth.detach().cpu().numpy()[0,0,:,:])

            axarr[1].set_title('Ground truth labels')

            plt.show()

            plt.gcf().show()

            running_loss = 0.0

    print(f'Epoch {epoch+1}, loss: ',np.mean(epoch_loss))