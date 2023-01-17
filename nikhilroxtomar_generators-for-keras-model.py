# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import numpy as np
import cv2
import keras
from skimage.transform import resize
class Generator(keras.utils.Sequence):
    def __init__(self, folder, files, batch_size=32, image_size=101, augment=False, mode='train'):
        self.folder = folder
        self.x_files = files
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.mode = mode
        self.on_epoch_end()

    def __load__(self, filename):
        if self.mode == 'train':
            image_path = self.folder + '/images/' + filename
            mask_path  = self.folder + '/masks/' + filename

            # Reading the image
            image = cv2.imread(image_path, 1)
            mask  = cv2.imread(mask_path, 0)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            # Normalizing the image
            image = image/255.0
            mask  = mask/255.0

            # Resizing the image
            if image.shape[0] != self.image_size:
                image = resize(image, (self.image_size, self.image_size, 3), mode='constant', preserve_range=True)
                mask  = resize(mask, (self.image_size, self.image_size, 1), mode='constant', preserve_range=True)
            return image, mask
            
        elif self.mode == 'test':
            image_path = self.folder + '/images/' + filename
            image = cv2.imread(image_path, 1)
            image = image/255.0
            if image.shape[0] != self.image_size:
                image = resize(image, (self.image_size, self.image_size, 3), mode='constant', preserve_range=True)
            return image


    def __getitem__(self, index):
        # Select batch
        if (index+1)*self.batch_size > len(self.x_files):
            self.batch_size = len(self.x_files) - index*self.batch_size
        files_batch = self.x_files[index*self.batch_size:(index+1)*self.batch_size]

        if self.mode == 'train':
            # Loading images and masks
            image = []
            mask  = []

            for filename in files_batch:
                tmp_img, tmp_mask = self.__load__(filename)
                image.append(tmp_img)
                mask.append(tmp_mask)

            image = np.array(image)
            mask  = np.array(mask)
            return image, mask

        elif self.mode == 'test':
            image = []
            for i, filename in enumerate(files_batch):
                tmp_img = self.__load__(filename)
                image.append(tmp_img)

            image = np.array(image)
            return image

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.x_files) / float(self.batch_size)))
# Training
folder = '../input/train/'
x_folder = '../input/train/images/'
y_folder = '../input/train/masks/'

x_files = os.listdir(x_folder)
y_files = os.listdir(y_folder)

n_valid_samples = 800

train_x_files = x_files[n_valid_samples:]
train_y_files = y_files[n_valid_samples:]
train_files = train_x_files

valid_x_files = x_files[:n_valid_samples]
valid_y_files = y_files[:n_valid_samples]
valid_files = valid_x_files

print("Training Mode")
gen = Generator(folder, train_files)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)
x, y = gen.__getitem__(1)
print(x.shape, y.shape)
x, y = gen.__getitem__(2)
print(x.shape, y.shape)
#Testing
folder = '../input/test/'
x_folder = '../input/test/images/'
test_x_files = os.listdir(x_folder)
test_files = test_x_files

print("Test Mode")
gen = Generator(folder, test_files, mode='test')
x = gen.__getitem__(0)
print(x.shape)
x = gen.__getitem__(1)
print(x.shape)
x = gen.__getitem__(2)
print(x.shape)
