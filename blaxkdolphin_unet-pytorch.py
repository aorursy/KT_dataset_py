import os

import time

import math

import glob

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as T



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)





#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



model_path = 'unet_model.pth'
! unzip -q /kaggle/input/tgs-salt-identification-challenge/competition_data.zip
image_path = "/kaggle/working/competition_data/train/images"

mask_path = "/kaggle/working/competition_data/train/masks"
names = ['6caec01e67','2bfa664017','1544a0e952']

images = [Image.open(os.path.join(image_path, name+'.png')) for name in names]

masks = [Image.open(os.path.join(mask_path, name+'.png')) for name in names]



transforms = T.Compose([T.Grayscale(), T.ToTensor()])

x = torch.stack([transforms(image) for image in images])

y = torch.stack([transforms(mask) for mask in masks])



fig = plt.figure( figsize=(9, 9))



ax = fig.add_subplot(331)

plt.imshow(images[0])

ax = fig.add_subplot(332)

plt.imshow(masks[0])

ax = fig.add_subplot(333)

ax.imshow(x[0].squeeze(), cmap="Greys")

ax.imshow(y[0].squeeze(), alpha=0.5, cmap="Greens")



ax = fig.add_subplot(334)

plt.imshow(images[1])

ax = fig.add_subplot(335)

plt.imshow(masks[1])

ax = fig.add_subplot(336)

ax.imshow(x[1].squeeze(), cmap="Greys")

ax.imshow(y[1].squeeze(), alpha=0.5, cmap="Greens")



ax = fig.add_subplot(337)

plt.imshow(images[2])

ax = fig.add_subplot(338)

plt.imshow(masks[2])

ax = fig.add_subplot(339)

ax.imshow(x[2].squeeze(), cmap="Greys")

ax.imshow(y[2].squeeze(), alpha=0.5, cmap="Greens")



plt.show()
class segmentDataset(Dataset):

    def __init__(self, image_path, mask_path):

        self.image_path = image_path

        self.mask_path = mask_path

        

        image_list= glob.glob(image_path +'/*.png')

        sample_names = []

        for file in image_list:

            sample_names.append(file.split('/')[-1].split('.')[0])

            

        self.sample_names = sample_names

        

        self.transforms = T.Compose([T.Grayscale(), T.ToTensor()])

            

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.image_path, self.sample_names[idx]+'.png') )

        mask = Image.open(os.path.join(self.mask_path, self.sample_names[idx]+'.png') )

        return self.transforms(image), self.transforms(mask)



    def __len__(self):

        return len(self.sample_names)
train_dataset = segmentDataset(image_path, mask_path)
max_images = 64

grid_width = 8

grid_height = int(max_images / grid_width)

fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*2, grid_height*2))

for i, idx in enumerate(range(max_images)):

    image, mask = train_dataset[idx]

    

    ax = axs[int(i / grid_width), i % grid_width]

    ax.imshow(image.squeeze(), cmap="Greys")

    ax.imshow(mask.squeeze(), alpha=0.5, cmap="Greens")

   

    ax.set_yticklabels([])

    ax.set_xticklabels([])

plt.show()
class convBlock(nn.Module):

    def __init__(self, in_channels, filters, size, stride = 1, activation = True):

        super(convBlock, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_channels, filters, size, stride = stride, padding = size//2)

        self.norm = nn.BatchNorm2d(filters)



    def forward(self, x):

        x = self.conv(x)

        x = self.norm(x)

        if self.activation:

            return F.relu(x)

        else:

            return x

    

class residualBlock(nn.Module):

    def __init__(self, in_channels, filters, size = 3):

        super(residualBlock, self).__init__()



        self.norm = nn.BatchNorm2d(in_channels)

        self.conv1 = convBlock(in_channels, filters, size)

        self.conv2 = convBlock(filters, filters, size, activation=False)



    def forward(self, x):

        residual = x  

        x = F.relu(x)

        x = self.norm(x)

        x = self.conv1(x)

        x = self.conv2(x)

        #x += residual

        return x 

    

class deconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 2, stride = 2):

        super(deconvBlock, self).__init__()

        

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)



    def forward(self, x1, x2):

        xd = self.deconv(x1)

        x = torch.cat([xd, x2], dim = 1)

        return x
class UnetModel(nn.Module):



    def __init__(self, filters = 16, dropout = 0.5):

        super(UnetModel, self).__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, filters, 3, padding = 1),

            residualBlock(filters, filters),

            residualBlock(filters, filters),

            nn.ReLU()

        )

        

        self.conv2 = nn.Sequential(

            nn.MaxPool2d(2, 2),

            nn.Dropout(dropout/2),

            nn.Conv2d(filters, filters * 2, 3, padding = 1),

            residualBlock(filters * 2, filters * 2),

            residualBlock(filters * 2, filters * 2),

            nn.ReLU()

        )

        

        self.conv3 = nn.Sequential(

            nn.MaxPool2d(2, 2),

            nn.Dropout(dropout),

            nn.Conv2d(filters * 2, filters * 4, 3, padding = 1),

            residualBlock(filters * 4, filters * 4),

            residualBlock(filters * 4, filters * 4),

            nn.ReLU()

        )

        

        self.conv4 = nn.Sequential(

            nn.MaxPool2d(2, 2),

            nn.Dropout(dropout),

            nn.Conv2d(filters * 4, filters * 8, 3, padding = 1),

            residualBlock(filters * 8, filters * 8),

            residualBlock(filters * 8, filters * 8),

            nn.ReLU()

        )

            



        self.middle = nn.Sequential(

            nn.MaxPool2d(2, 2),

            nn.Dropout(dropout),

            nn.Conv2d(filters * 8, filters * 16, 3, padding = 3//2),

            residualBlock(filters * 16, filters * 16),

            residualBlock(filters * 16, filters * 16),

            nn.ReLU()

        )

        

        self.deconv4 = deconvBlock(filters * 16, filters * 8, 2)

        self.upconv4 = nn.Sequential(

            nn.Dropout(dropout),

            nn.Conv2d(filters * 16, filters * 8, 3, padding = 1),

            residualBlock(filters * 8, filters * 8),

            residualBlock(filters * 8, filters * 8),

            nn.ReLU()

        )

  



        self.deconv3 = deconvBlock(filters * 8, filters * 4, 3)

        self.upconv3 = nn.Sequential(

            nn.Dropout(dropout),

            nn.Conv2d(filters * 8, filters * 4, 3, padding = 1),

            residualBlock(filters * 4, filters * 4),

            residualBlock(filters * 4, filters * 4),

            nn.ReLU()

        )

        

        self.deconv2 = deconvBlock(filters * 4, filters * 2, 2)

        self.upconv2 = nn.Sequential(

            nn.Dropout(dropout),

            nn.Conv2d(filters * 4, filters * 2, 3, padding = 1),

            residualBlock(filters * 2, filters * 2),

            residualBlock(filters * 2, filters * 2),

            nn.ReLU()

        )



        self.deconv1 = deconvBlock(filters * 2, filters, 3)

        self.upconv1 = nn.Sequential(

            nn.Dropout(dropout),

            nn.Conv2d(filters * 2, filters, 3, padding = 1),

            residualBlock(filters, filters),

            residualBlock(filters, filters),

            nn.ReLU(),

            nn.Dropout(dropout/2),

            nn.Conv2d(filters, 1, 3, padding = 1)

        )



    def forward(self, x):

        conv1 = self.conv1(x) 

        # 101 -> 50

        conv2 = self.conv2(conv1) 

        # 50 -> 25

        conv3 = self.conv3(conv2) 

        # 25 -> 12

        conv4 = self.conv4(conv3) 

        # 12 - 6

        x = self.middle(conv4) 

        

        # 6 -> 12

        x = self.deconv4(x, conv4)

        x = self.upconv4(x)

        # 12 -> 25

        x = self.deconv3(x, conv3)

        x = self.upconv3(x)

        # 25 -> 50

        x = self.deconv2(x, conv2)

        x = self.upconv2(x)

        # 50 -> 101

        x = self.deconv1(x, conv1)

        x = self.upconv1(x)



        return x
def get_iou_score(outputs, labels):

    A = labels.squeeze().bool()

    pred = torch.where(outputs<0., torch.zeros_like(outputs), torch.ones_like(outputs))

    B = pred.squeeze().bool()

    intersection = (A & B).float().sum((1,2))

    union = (A| B).float().sum((1, 2)) 

    iou = (intersection + 1e-6) / (union + 1e-6)  

    return iou

  

def train_one_batch(model, x, y):

    x, y = x.to(device), y.to(device)



    outputs = model(x)

    loss = loss_fn(outputs, y)

    iou = get_iou_score(outputs, y).mean()

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item(), iou.item()

NUM_EPOCHS = 200

BATCH_SIZE = 64



model = UnetModel().to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters())

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')



loss_fn = nn.BCEWithLogitsLoss()



train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

steps  = train_dataset.__len__()// BATCH_SIZE

print(steps,"steps per epoch")



start = time.time()

train_losses = []

train_ious = []

for epoch in range(1, NUM_EPOCHS + 1):

    print('-' * 10)

    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))

    running_iou = []

    running_loss = []

    for step, (x, y) in enumerate(train_dataloader):

        loss, iou = train_one_batch(model, x, y)

        running_iou.append(iou)

        running_loss.append(loss)

        print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}'.format(100*(step+1)/steps, loss,iou), end = "") 

        

    print('\r{:6.1f} %\tloss {:8.4f}\tIoU {:8.4f}\t{}'.format(100*(step+1)/steps,np.mean(running_loss),np.mean(running_iou), timeSince(start)))

    scheduler.step(np.mean(running_iou))

    

    train_losses.append(loss)

    train_ious.append(iou)
plt.plot(train_losses, label = 'loss')

plt.plot(train_ious, label = 'IoU')

plt.xlabel('Epoch')

plt.ylabel('Metric')

plt.legend()

plt.show()
## save weights    

torch.save(model.cpu().state_dict(), model_path)



model.eval()
names = ['6caec01e67','2bfa664017','1544a0e952']

images = [Image.open(os.path.join(image_path, name+'.png')) for name in names]

masks = [Image.open(os.path.join(mask_path, name+'.png')) for name in names]



transforms = T.Compose([T.Grayscale(), T.ToTensor()])

x = torch.stack([transforms(image) for image in images])

y = torch.stack([transforms(mask) for mask in masks])



outputs = model(x)

preds = torch.where(outputs<0., torch.zeros_like(outputs), torch.ones_like(outputs))

ious = get_iou_score(outputs, y)



fig = plt.figure( figsize=(9, 12))



ax = fig.add_subplot(331)

plt.imshow(images[0])

ax = fig.add_subplot(332)

ax.imshow(x[0].squeeze(), cmap="Greys")

ax.imshow(y[0].squeeze(), alpha=0.5, cmap="Greens")

ax = fig.add_subplot(333)

ax.imshow(x[0].squeeze(), cmap="Greys")

ax.imshow(preds[0].squeeze(), alpha=0.5, cmap="OrRd")

ax.set_title("IoU: " + str(round(ious[0].item(), 2)), loc = 'left')



ax = fig.add_subplot(334)

plt.imshow(images[1])

ax = fig.add_subplot(335)

ax.imshow(x[1].squeeze(), cmap="Greys")

ax.imshow(y[1].squeeze(), alpha=0.5, cmap="Greens")

ax = fig.add_subplot(336)

ax.imshow(x[1].squeeze(), cmap="Greys")

ax.imshow(preds[1].squeeze(), alpha=0.5, cmap="OrRd")

ax.set_title("IoU: " + str(round(ious[1].item(), 2)), loc = 'left')



ax = fig.add_subplot(337)

plt.imshow(images[2])

ax = fig.add_subplot(338)

ax.imshow(x[2].squeeze(), cmap="Greys")

ax.imshow(y[2].squeeze(), alpha=0.5, cmap="Greens")

ax = fig.add_subplot(339)

ax.imshow(x[2].squeeze(), cmap="Greys")

ax.imshow(preds[2].squeeze(), alpha=0.5, cmap="OrRd")

ax.set_title("IoU: " + str(round(ious[2].item(), 2)), loc = 'left')



plt.show()
trainiter = iter(train_dataloader)

images, masks = next(trainiter)



outputs = model(images)

preds = torch.where(outputs<0., torch.zeros_like(outputs), torch.ones_like(outputs))

ious = get_iou_score(outputs, masks).numpy()
max_images = outputs.size(0)

grid_width = 8

grid_height = int(max_images / grid_width)

fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*2, grid_height*2))

for i, _data in enumerate(zip(images, masks, preds, ious)):

    image, mask, pred, iou = _data

    

    ax = axs[int(i / grid_width), i % grid_width]

    

    ax.imshow(image.squeeze(), cmap = "Greys")

    ax.imshow(mask.squeeze(), alpha = 0.5, cmap = "Greens")

    ax.imshow(pred.squeeze(), alpha = 0.3, cmap = "OrRd")

    ax.set_title("IoU: " + str(round(iou, 2)), loc = 'left')

    

    ax.set_yticklabels([])

    ax.set_xticklabels([])

plt.suptitle("Green: salt, Red: prediction")

plt.show()
## https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks



def rle_encode(im):

    '''

    im: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = im.flatten(order = 'F')

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



transforms = T.Compose([T.Grayscale(), T.ToTensor()])
image_path = "/kaggle/working/competition_data/test/images"

sub_df = pd.read_csv('/kaggle/working/competition_data/sample_submission.csv')

n = sub_df.shape[0]



rle_mask = []

for idx in range(n):

    ## load image

    sample_name = sub_df['id'][idx]

    image = Image.open(os.path.join(image_path, sample_name+'.png') )

    image = transforms(image)

    ## predict

    out = model(image.unsqueeze(0)).squeeze()

    pred = torch.where(out<0., torch.zeros_like(out), torch.ones_like(out))

    ## write mask

    rle_mask.append(rle_encode(pred.numpy()))

    print("\rprogress {}/{}".format(idx+1, n), end = "")

    

sub_df['rle_mask'] = rle_mask
sub_df.to_csv('submission.csv', index = False)