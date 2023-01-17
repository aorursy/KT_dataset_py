import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# print out the names of the first 2 image_files (total = 4 images for train_imgaes & train_label_masks) with the train, test, submission.csv files & 5 file.hdf5
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:2]:
        print(os.path.join(dirname, filename))
import cv2
import openslide
import skimage.io
import random
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

BASE_PATH = '../input/prostate-cancer-grade-assessment'
data_dir = f'{BASE_PATH}/train_images'
mask_dir = f'{BASE_PATH}/train_label_masks'
hdf5_dir = r'/kaggle/input/radboud-database/radboud_tiles_coordinates.h5'
import deepdish as dd

df = dd.io.load(hdf5_dir)
len(df)//36, len(df[0]), df[0], len(df)
def load_data_and_mask(ID, coordinates, level = 1):
    """
    Input args:
        ID (str): img_id from the dataset
        coordinates (list of int): list of coordinates, includes: [x_start, x_end, y_start, y_end] from h5.database
        level (={0, 1, 2}) : level of images for loading with skimage
    Return: 3D tiles shape 512x512 of the mask images and data images w.r.t the input_coordinates, ID and level
    """
    data_img = skimage.io.MultiImage(os.path.join(data_dir, f'{ID}.tiff'))[level]
    mask_img = skimage.io.MultiImage(os.path.join(mask_dir, f'{ID}_mask.tiff'))[level]
    coordinates = [coordinate // 2**(2*level) for coordinate in coordinates]
    data_tile = data_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3], :]
    mask_tile = mask_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3], :]
    data_tile = cv2.resize(data_tile, (512, 512))
    mask_tile = cv2.resize(mask_tile, (512, 512))
    del data_img, mask_img
    
    # Load and return small image
    return data_tile, mask_tile
from torch.utils.data import Dataset, DataLoader
import torch

class PANDADataset(Dataset):
    def __init__(self, df, level = 2, transform=None):
        self.df = df
        self.level = level
        self.transform = transform

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index, level = 2):
        ID = self.df[index][0]
        coordinate = self.df[index][1: ]
        image, mask = load_data_and_mask(ID, coordinate, level)
        
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask[:,:,0])
    
cls = PANDADataset(df, 1)
%time cls[0][0].size(), cls[0][1].size(), len(cls)
plt.imshow(cls[300][1])
dataLoader = DataLoader(cls, batch_size=8, shuffle=True, num_workers=8)

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
# --- Unet params
n_classes= 6    # number of classes in the data mask that we'll aim to predict


in_channels = 3  # input channel of the data, RGB = 3
padding = True   # should levels be padded
depth = 5        # depth of the network 
wf = 2           # wf (int): number of filters in the first layer is 2**wf, was 6
up_mode = 'upconv' #should we simply upsample the mask, or should we try and learn an interpolation 
batch_norm = True #should we use batch normalization between the layers

# --- training params

batch_size = 8
patch_size = 512
num_epochs = 1
edge_weight = 1.1 # edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter boosts their values along the lines of the original UNET paper
phases = ["train","val"] # how many phases did we create databases for?
validation_phases= ["val"] # when should we do valiation? note that validation is time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
gpuid = 0
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')
model = UNet(n_classes = n_classes, in_channels = in_channels, 
             padding = padding, depth = depth, wf = wf, 
             up_mode = up_mode, batch_norm = batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

optim = torch.optim.Adam(model.parameters()) #adam is going to be the most robust
criterion = nn.CrossEntropyLoss()
import time
print('============================================== Training started ==============================================')
for epoch in range(num_epochs):
    print('==============================================================================================================')
    # model.train()  # Set model to training mode
    running_loss = 0.0
    train_accuracy = 0
    total_train = 0
    correct_train = 0
    t0 = time.time()
    
    for i, data in enumerate(dataLoader, 0):
        inputs, labels = data
        inputs = inputs.to(device,dtype = torch.float) 
        labels = labels.to(device,dtype = torch.int64)
        
        # zero the parameter gradients
        optim.zero_grad()
        
        # =========================== forward + backward + optimize ===========================
        outputs = model(inputs)
        #_, outputs = torch.max(model(inputs), axis = 1)
        #outputs = torch.argmax(model(inputs), axis = 1)
        
        ## =========================== Loss computation ===========================
        loss = criterion(outputs, labels)
        loss.sum().backward()
        optim.step()       
        
        ## =========================== Accuracy computation ========================================        
        # return the indices of max values along rows in softmax probability output
        predicted = torch.argmax(outputs, axis = 1)
        
        # number of pixel in the batch
        total_train += labels.nelement()        
        # count of the number of times the neural network has produced a correct output, and 
        # we take an accumulating sum of these correct predictions so that we can determine the accuracy of the network.
        #print(labels == predicted)
        correct_train += (labels == predicted).sum().item()        
        
        # =========================== print statistics ===========================
        running_loss += loss.mean()
        
        train_accuracy = correct_train / total_train
        
        if i % 300 == 299:    # print every 2000 mini-batches
            t1 = time.time()
            h = (t1 - t0) // 3600
            m = (t1 - t0 - h*3600) // 60
            s = (t1 - t0) % 60
            print('Eps %02d, upto %05d mnbch; after %02d (hours) %02d (minutes) and %02d (seconds);  train_loss = %.3f, train_acc = %.3f'%
                  (epoch + 1, i + 1, h, m, s, running_loss / 300, train_accuracy))
            running_loss = 0.0
print('==============================================================================================================')
print('============================================== Finished Training =============================================')
df = dd.io.load(hdf5_dir)
cls_test = PANDADataset(df[ : 800], 1)

plt.figure(figsize = (20, 10))
for k in range(5):
    idx = np.random.randint(0, 800)
    a = cls_test[idx][0].permute((1, 2, 0)).detach().squeeze().cpu().numpy()
    b = cls_test[idx][1].detach().squeeze().cpu().numpy()
    cmap =  matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    plt.subplot(2, 5, k+1), plt.imshow(a)
    plt.subplot(2, 5, k+6), plt.imshow(b, cmap = cmap)
plt.show()
print(a.min(), a.max(), b.min(), b.max())
data_inp = []
predicts = []
true_mask = []
cls_test = PANDADataset(df[ : 80], 1)
dataLoader_test = DataLoader(cls_test, batch_size=8, shuffle=True, num_workers=8)
for i, data in enumerate(dataLoader_test, 0):
    inputs, labels = data
    inputs = inputs.to(device,dtype = torch.float) 
    labels = labels.to(device,dtype = torch.int64) ## type('torch.LongTensor').to(device)
    
    predict = torch.argmax(model(inputs), axis = 1)
    
    ## append
    predicts += predict
    data_inp += inputs
    true_mask += labels
    ## freeze
    del inputs, labels, predict, data
    
print(len(predicts), predicts[0].shape)
print(len(data_inp), data_inp[0].shape)
print(len(true_mask), true_mask[0].shape)
fig, ax = plt.subplots(1, 4, figsize=(25, 5.5))
for k in range(4):
    c = predicts[k].detach().squeeze().cpu().numpy()
    ax[k].imshow(c, cmap = cmap), ax[k].set_title('predict %s'%(k+1))
plt.show()

fig, ax = plt.subplots(1, 4, figsize=(25, 5.5))
for k in range(4):
    d = true_mask[k].detach().squeeze().cpu().numpy()
    ax[k].imshow(d, cmap = cmap), ax[k].set_title('true_mask %s'%(k+1))
plt.show()
#plt.imshow(torch.argmax(predicts[0], axis = 0).detach().squeeze().cpu().numpy())
torch.argmax(predicts[0], axis = 0)[0]
ID_list = os.listdir(mask_dir)
ID_list = [u.replace('_mask.tiff', '') for u in ID_list]

ID = ID_list[0]
def split_get_tiles(img_id, crit = 0.0005, size=512, n_tiles=36):    
    """
    ==================================================================================================
    Input:  img_id (str): image_id from the train dataset, such as '004dd32d9cd167d9cc31c13b704498af'  
            crit (float) in (0, 1): the proportion of the dark_region over whole image (size 256 x 256)
            size (int) : image size
            n_tiles : number of tiles
    return: 
            list of (img_id, x_start, x_end, y_start, y_end) images size 512x512    
            ==========================================================================================
    writen by Nhan
    ==================================================================================================
    """
    img = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))[0]
    tile_size = 512
    h, w = img.shape[: 2]
    nc = int(w / 512)
    nr = int(h / 512)
    img_ls = []
    coord_ls = []
    S_img_tile = 512*512*3
    
    for i in range(nr):
        for j in range(nc):
            x_start, y_start = int(i*512), int(j*512)
            image_dt = img[ x_start : x_start + 512, y_start : y_start + 512 , :]
            if (image_dt.min() < 185):
                count = len(image_dt[image_dt <= 121])
                if count/(S_img_tile) >= crit:
                    image_dt = cv2.resize(image_dt, (size, size), interpolation = cv2.INTER_AREA)
                    img_ls.append(image_dt)
                    del image_dt, x_start, y_start

    ## choose n_tiles image has a best-view_range 
    img3_dt_ = np.array(img_ls)
    idxs_dt_ = np.argsort(img3_dt_.reshape(img3_dt_.shape[0],-1).sum(-1))[:n_tiles]
    
    ## attach
    list_image = []
    for final_index in idxs_dt_:
        list_image.append(img_ls[final_index])
    for i in range(8): 
        yield list_image[i: i+8]
# Function to make tiles from one image
# ...

# Funtion to calculate the isup from the list of outputs.
def ISUP(result):
    # result: a list of masks
    # Translation matrix of gleason scores to isup
    import numpy as np
    isup_mat = np.array([[1, 2, 4],[3, 4, 5],[4, 5, 5]])
    # calculate the most dominant gleason score
    p = np.zeros(6)
    for mask in result:
        for i in range(3, 6):
            p[i] += len(np.nonzeros((mask==i)*1)) 
    gscore1 = max(argmax(p), 3)
    p[gscore1] = 0
    if argmax(p) ==0:
        gscore2 = gscore1
    else:
        gscore2 = max(argmax(p), 3)
        
    return isup_mat[gscore1-3, gscore2-3]
# Configuration
import openslide
img_id = '6d1a11077fe4183a4109d649cf319923'
# Load image
#osh = openslide.OpenSlide(file)

# Load model
model = UNet(n_classes = n_classes, in_channels = in_channels, 
             padding = padding, depth = depth, wf = wf, 
             up_mode = up_mode, batch_norm = batch_norm)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

# Define and add device
gpuid = 0
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')
    
model.to(device)

# Evaluation mode
model.eval()

# Cut image into tiles
tiles = split_get_tiles(img_id, crit = 0.0005, size=512, n_tiles=36)

result = []
# Make masks
for inputs in tiles: 
    predicts = torch.argmax(model(inputs), axis = 1)
    result = result.append(predicts)

# Make ISUP grade
isup = ISUP(result)

