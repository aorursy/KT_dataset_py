import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd

import torch

from torch.utils import data
class TGSSaltDataset(data.Dataset):
    
    def __init__(self, root_path, file_list):
        self.root_path = root_path
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        image = np.array(imageio.imread(image_path), dtype=np.uint8)
        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        
        return image, mask, file_id
depths_df = pd.read_csv('../input/depths.csv')

train_path = "../input/train/"
train_df = pd.read_csv('../input/train.csv')
train = pd.merge(train_df, depths_df, on='id', how='inner')
file_list = list(train['id'].values)
depth_list = list(train['z'].values)
train.head()
dataset = TGSSaltDataset(train_path, file_list)
train = pd.merge(train_df, depths_df, on='id', how='inner')
MaskArea_list = []
for i in range(len(train['rle_mask'])):
    elements = str(train['rle_mask'][i]).split(' ')
    if len(elements) > 1:
        area = sum([int(elements[i]) for i in range(len(elements)) if i%2 == 1])
    else:
        area = 0
    MaskArea_list.append(area)


def plot2x2Array(image, mask, Image_title = None, Mask_title=None):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title(Image_title)
    axarr[1].set_title(Mask_title)
#for i in range(len(dataset)):
for i in range(10):
    image, mask, file_id = dataset[i]
    plot2x2Array(image, mask, Image_title = file_id, Mask_title = file_id)
    print('Image %d / %d' %((i+1), len(dataset)))
    print('depth of id (%s): %d' %(file_list[i], depth_list[i]))
    print('Mask area: %d' %MaskArea_list[i])
    plt.show()
#len(depth_list)
plt.plot(MaskArea_list, depth_list, 'b.')
plt.xlabel('Mask area')
plt.ylabel('Depth')
plt.show()