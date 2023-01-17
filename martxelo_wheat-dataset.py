# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from scipy.ndimage import zoom

import ast

from tqdm import tqdm
ORIG_SIZE = 1024

SCALE = 4

IMG_SIZE = ORIG_SIZE//SCALE



def get_mask(fname, train):

    '''

    Returns an image with ones inside bboxes and zeros outside

    '''

    

    fname = fname.split('.')[0]

    train = train[train['image_id']==fname]

    

    boxes = train['bbox'].tolist()

    

    mask = np.zeros((IMG_SIZE,IMG_SIZE))

    

    for box in boxes:

        box = ast.literal_eval(box)

        box = [int(i) for i in box]

        

        x1, x2 = box[1]//SCALE, (box[1] + box[3])//SCALE

        y1, y2 = box[0]//SCALE, (box[0] + box[2])//SCALE

        

        mask[x1:x2,y1:y2] = 1

        

    return mask

        



def create_dataset():

    

    images = os.listdir('/kaggle/input/global-wheat-detection/train')

    train = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

    

    N = len(images)

    x = np.zeros((N,IMG_SIZE,IMG_SIZE,3))

    y = np.zeros((N,IMG_SIZE,IMG_SIZE,1))

    

    for i, fname in enumerate(tqdm(images)):

        

        img = mpimg.imread('/kaggle/input/global-wheat-detection/train/'+ fname)/255

        img = zoom(img, (1/SCALE,1/SCALE,1))

        x[i,:,:,:] = img

        

        mask = get_mask(fname, train)

        y[i,:,:,:] = np.expand_dims(mask, axis=2)

    

    # because of zoom some values are lower than zero or higher than one.

    x = (x - x.min())/(x.max() - x.min())

    

    # save some memory

    x, y = np.float32(x), np.float32(y)

    

    return x, y

    



images, masks = create_dataset()





np.save('/kaggle/working/images.npy', images)

np.save('/kaggle/working/masks.npy', masks)

    

print('images.shape=', images.shape)

print('masks.shape=', masks.shape)

print('images.max=', images.max())

print('masks.max=', masks.max())

print('images.min=', images.min())

print('masks.min=', masks.min())
# plot a random figure figure with its masks

n = np.random.randint(low=0, high=3422)



fig = plt.figure(figsize=(12,12))

plt.imshow(images[n] + masks[n]*0.3)