# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import getcwd
import shutil
import zipfile
from shutil import copyfile
import random
print(os.listdir("../input"))
cat_dog_path = "../input/dogs-vs-cats"
os.listdir(cat_dog_path)
train_zip = "../input/dogs-vs-cats/train.zip"
zip_file = zipfile.ZipFile(train_zip,'r')
zip_file.extractall('.')
zip_file.close()
test_zip = "../input/dogs-vs-cats/test1.zip"
zip_file = zipfile.ZipFile(train_zip,'r')
zip_file.extractall('.')
zip_file.close()
try:
    main_dir = "/kaggle/working/"
    train_dir = "train"
    val_dir = "val"
    train_dir = os.path.join(main_dir,train_dir)
    
    # Directory with our training cat/dog pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
    
    # Directory with our validation cat/dog pictures
    val_dir = os.path.join(main_dir,"val")
    os.mkdir(val_dir)
    val_cats_dir = os.path.join(val_dir, 'cats')
    val_dogs_dir = os.path.join(val_dir, 'dogs')
    os.mkdir(val_cats_dir)
    os.mkdir(val_dogs_dir)

except OSError:
    pass
main_dir = "/kaggle/working/"
train_dir = "train"
train_path = os.path.join(main_dir,train_dir)

prefixed_dogs = [filename for filename in os.listdir(train_path) if filename.startswith("dog.")]
print(len(prefixed_dogs))
prefixed_cats = [filename for filename in os.listdir(train_path) if filename.startswith("cat.")]
print(len(prefixed_cats))

def move_files(src_file):
    
    for filename in prefixed_dogs:
        shutil.move(src_file+filename, src_file+'dogs/'+filename)
        
    for filename in prefixed_cats:
        shutil.move(src_file+filename, src_file+'cats/'+filename)
    

move_files("/kaggle/working/train/")
print(len(os.listdir('/kaggle/working/train/dogs')))
print(len(os.listdir('/kaggle/working/train/cats')))
print(len(os.listdir('/kaggle/working/train')))
def split_data(SOURCE, VALID, SPLIT_SIZE):
# This funtion takes as argument:
###SOURCE : the directory's path of images that will be splitted
###VALID : the directory's path of the validation receiving the dogs or the cats images
###SPLIT_SIZE: the size of the split. 0.9 means 90% of cats images will remain in train/cats and 10% will be moved to the validation directory's cats 
###and the same will be done to the dogs images
    SRC_files = [f for f in os.listdir(SOURCE) if os.path.isfile(os.path.join(SOURCE, f))]
    SRC_Size = len(SRC_files)
    #print(SRC_Size)
    if SRC_Size != 0:
        # we shuffle the images before the split
        shuffled_files = random.sample(SRC_files, len(SRC_files))
        #print("shuffled")
        TRN_size = int(SRC_Size * SPLIT_SIZE)
        VAL_SIZE = int(SRC_Size - TRN_size)
        print(TRN_size)
        train_set = shuffled_files[0:TRN_size]
        val_set = shuffled_files[-VAL_SIZE:SRC_Size]
        for filename in val_set:
            if os.path.getsize(SOURCE+filename)!=0:
                shutil.move(SOURCE+filename, VALID+filename)
            else:
                print(filename + ' is zero length. So ignoring!')
                pass


                    
CAT_SOURCE_DIR = "/kaggle/working/train/cats/"
TESTING_CATS_DIR = "/kaggle/working/val/cats/"

DOG_SOURCE_DIR = "/kaggle/working/train/dogs/"
TESTING_DOGS_DIR = "/kaggle/working/val/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('/kaggle/working/train/dogs')))
print(len(os.listdir('/kaggle/working/train/cats')))
print(len(os.listdir('/kaggle/working/train')))
print(len(os.listdir('/kaggle/working/val/dogs')))
print(len(os.listdir('/kaggle/working/val/cats')))
from matplotlib.image import imread
sample_dog_image = '/kaggle/working/train/dogs/dog.7024.jpg'
plt.imshow(imread(sample_dog_image))
train_path = "/kaggle/working/train/"
test_path="/kaggle/working/test/"
image_width = 128
image_height = 128
image_color = 3
image_shape = (image_width,image_height,image_color)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from numpy import expand_dims
sample_img_gen = ImageDataGenerator(rotation_range=20,
                                    height_shift_range=0.06,
                                    width_shift_range=0.06,
                                    shear_range=0.1,
                                    horizontal_flip=True,
                                    zoom_range=[0.5,0.8])

sample_image = load_img(sample_dog_image)
sample_image_array = img_to_array(sample_image)
test_image = expand_dims(sample_image_array,0)
image_aug = sample_img_gen.flow(test_image,batch_size=1)
for i in range(9):
    plt.subplot(330+1+i)
    batch = image_aug.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()
for i in range(9):
    plt.subplot(330+1+i)
    batch = image_aug.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()
