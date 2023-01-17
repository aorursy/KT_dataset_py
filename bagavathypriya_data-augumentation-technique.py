# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile

zf= zipfile.ZipFile("../input/dogs-vs-cats/train.zip")
zf.filelist
zf.extractall()
img=zf.extract("train/dog.66.jpg")
y=plt.imread(img)
plt.imshow(y)
from keras.preprocessing.image import load_img, img_to_array,array_to_img,ImageDataGenerator
img=load_img("train/dog.66.jpg")
x=img_to_array(img)
print(x.shape)
x=x.reshape((1,)+x.shape)
print(x.shape)
datagen=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,horizontal_flip=True,rotation_range=40,
                           width_shift_range=0.2,height_shift_range=0.2,
                          fill_mode='nearest')

i = 0
for batch in datagen.flow(x, batch_size=1):
    i += 1
    if i > 20:
        break
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break