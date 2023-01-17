# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL

from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = '../input/siim-isic-melanoma-classification/'

TRAIN_IMAGE_PATH= DATA_PATH+'jpeg/train/'

TEST_IMAGE_PATH= DATA_PATH+'jpeg/test/'

RESIZE_DIM = 128



train = pd.read_csv(DATA_PATH+'train.csv')



def resize_image(path, image_name):

    image = PIL.Image.open(os.path.join(path,image_name+'.jpg'))

    image = image.resize((RESIZE_DIM,RESIZE_DIM), resample = PIL.Image.LANCZOS)

    return image
x_train = np.empty((train.shape[0], RESIZE_DIM,RESIZE_DIM,3), dtype=np.uint8)



for idx,image_name in enumerate(tqdm(train.image_name)):

    img_resized = resize_image(TRAIN_IMAGE_PATH,image_name)

    x_train[idx,:,:,:] = np.array(img_resized)



np.save('train128.npy', x_train)