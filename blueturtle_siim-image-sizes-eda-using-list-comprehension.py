# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import os

image_meta_path = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
image_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"
image_meta = pd.read_csv(image_meta_path)
image_meta["height"] = 0
image_meta["width"] = 0
from torch.utils.data import DataLoader, Dataset #Create an efficient dataloader set to feed images to the model

class TrainData(Dataset):

    def __init__(self, dataframe, image_dir, transforms = None):
        super().__init__()
        self.df = dataframe
        self.image_ids = dataframe['image_name'].unique()
        self.image_dir = image_dir


    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        height = image.shape[0]
        width = image.shape[1]
        
        return image_id, height, width
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]

train_dataset = TrainData(image_meta, image_path)

def sizefinder(image):
    image_id = image[0]
    height = image[1]
    width = image[2]
    image_meta.loc[image_meta['image_name'] == image_id, ['height']] = height
    image_meta.loc[image_meta['image_name'] == image_id, ['width']] = width
    
    
sizefinder = [sizefinder(image) for image in train_dataset]
image_meta.to_csv("ImageSizes.csv", index= False)
image_meta
import matplotlib.pyplot as plt

pd.DataFrame.hist(image_meta, column = "image_name", by = "height")
import seaborn as sns
image_meta = pd.read_csv("/kaggle/input/imagesizes/image_meta.csv")
height = sns.kdeplot(image_meta["height"])
width = sns.kdeplot(image_meta["width"])