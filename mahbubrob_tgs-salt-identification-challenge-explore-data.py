import os
import pathlib
import imageio
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
print('Parent Directory: ', os.listdir("../input"))
print('train Directory:  ', os.listdir("../input/train"))
print('test Directory:   ', os.listdir("../input/train"))
depths_df = pd.read_csv('../input/depths.csv')
train_df = pd.read_csv("../input/train.csv")
sample_submission_df = pd.read_csv("../input/sample_submission.csv")

train_df.head()
depths_df.head()

sample_submission_df.head()
class TGSSaltDataset(Dataset):
    
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
        
        return image, mask
depths_df = pd.read_csv('../input/train.csv')

train_path = "../input/train/"
file_list = list(depths_df['id'].values)
dataset = TGSSaltDataset(train_path, file_list)
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')
for i in range(5):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)