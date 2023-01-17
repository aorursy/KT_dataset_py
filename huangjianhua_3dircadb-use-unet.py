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
# -*- coding: utf-8 -*- 
# HDF5DatasetGenerator.py
import h5py
import os
import numpy as np

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        
        self.db = h5py.File(dbPath)
        self.numImages = self.db["images"].shape[0]
#        self.numImages = total
        print("total images:",self.numImages)
        self.num_batches_per_epoch = int((self.numImages-1)/batchSize) + 1
        
    
    def generator(self, shuffle=True, passes=np.inf):
        epochs = 0
        
        while epochs < passes:
            shuffle_indices = np.arange(self.numImages) 
            shuffle_indices = np.random.permutation(shuffle_indices)
            for batch_num in range(self.num_batches_per_epoch):
                
                start_index = batch_num * self.batchSize
                end_index = min((batch_num + 1) * self.batchSize, self.numImages)
                
                # h5py get item by index,参数为list，而且必须是增序
                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))
                
                images = self.db["images"][batch_indices,:,:,:]
                labels = self.db["masks"][batch_indices,:,:,:]
                
#                if self.binarize:
#                    labels = np_utils.to_categorical(labels, self.classes)
                
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    
                    images = np.array(procImages)
                
                if self.aug is not None:
                    # 不知道意义何在？本身images就有batchsize个了
                    (images, labels) = next(self.aug.flow(images, labels,
                                                        batch_size=self.batchSize))
                yield (images, labels)
            
            epochs += 1
            
    def close(self):
        self.db.close()
# -*- coding: utf-8 -*-
# class HDF5DatasetWriter

import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, image_dims, mask_dims, outputPath, bufSize=200):
        """
        Args:
        - bufSize: 当内存储存了bufSize个数据时，就需要flush到外存
        """
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already"
                             "exists and cannot be overwritten. Manually delete"
                             "the file before continuing", outputPath)
        
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("images", image_dims, dtype="float")
        self.masks = self.db.create_dataset("masks", mask_dims, dtype="int")
        
        self.bufSize = bufSize
        self.buffer = {"data": [], "masks": []}
        self.idx = 0
    

    def add(self, rows, masks):
        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)
        # 注意，用extend还有好处，添加的数据不会是之前list的引用！！
        self.buffer["data"].extend(rows)
        self.buffer["masks"].extend(masks)
        print("len ",len(self.buffer["data"]))
        
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
    
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i,:,:,:] = self.buffer["data"]
        self.masks[self.idx:i,:,:,:] = self.buffer["masks"]
        print("h5py have writen %d data"%i)
        self.idx = i
        self.buffer = {"data": [], "masks": []}
        
    
    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        
        self.db.close()
        return self.idx
# utils.py
import numpy as np
import cv2
import re
import sys

def get_pixels_hu(scans):
    #type(scans[0].pixel_array)
    #Out[15]: numpy.ndarray
    #scans[0].pixel_array.shape
    #Out[16]: (512, 512)
    # image.shape: (129,512,512)
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def transform_ctdata(image, windowWidth, windowCenter, normal=False):
        """
        注意，这个函数的self.image一定得是float类型的，否则就无效！
        return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5*float(windowWidth)
        newimg = (image - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg

def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    # 得到轴向上出现过目标（label>=1)的切片
    z = np.any(image, axis=(1,2)) # z.shape:(depth,)
    startposition,endposition = np.where(z)[0][[0,-1]]
    return startposition, endposition

def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    firstflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and firstflag:
            startposition = z
            firstflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

def clahe_equalized(imgs,start,end):
   assert (len(imgs.shape)==3)  #3D arrays
   #create a CLAHE object (Arguments are optional).
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   imgs_equalized = np.empty(imgs.shape)
   for i in range(start, end+1):
       imgs_equalized[i,:,:] = clahe.apply(np.array(imgs[i,:,:], dtype = np.uint8))
   return imgs_equalized

def get_highest_acc():
    pattern = 'model.(?P<epoch>\d+)-(?P<val_acc>[0-9]*\.?[0-9]*).hdf5'
    p = re.compile(pattern)
    acces = [float(p.match(f).groups()[1]) for f in os.listdir('models/') if p.match(f)]
    if len(acces) == 0:
        return sys.float_info.min
    else:
        return np.max(acces)
# generate_train.py
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
# from HDF5DatasetWriter import HDF5DatasetWriter
# from HDF5DatasetGenerator import HDF5DatasetGenerator
# from utils import *
from tqdm import tqdm
import cv2

# 可以在part1之前设定好（即循环外）
seed=1
data_gen_args = dict(rotation_range=3,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.01,
                    zoom_range=0.01,
                    fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
print('build ImageDataGenerator finished.')

# 可以在part1之前设定好（即循环外）
# 这儿的数量需要提前写好，感觉很不方便，但不知道怎么改，我是先跑了之前的程序，计算了一共有多少
# 张图片后再写的，但这样明显不是好的解决方案
outputPath = "./data_train/train_liver.h5"

if os.path.exists(outputPath):
  os.remove(outputPath)
dataset = HDF5DatasetWriter(image_dims=(2782, 512, 512, 1),
                            mask_dims=(2782, 512, 512, 1),
                            outputPath=outputPath)

print('build HDF5DatasetWriter finished')

#part1
for i in range(1,3): # 前17个人作为测试集
   full_images = [] # 后面用来存储目标切片的列表
   full_livers = [] #功能同上
   # 注意不同的系统，文件分割符的区别
   label_path = '../input/liver20/3Dircadb1/3Dircadb1.%d/MASKS_DICOM/liver'%i
   data_path = '../input/liver20/3Dircadb1/3Dircadb1.%d/PATIENT_DICOM'%i
   liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
   # 注意需要排序，即使文件夹中显示的是有序的，读进来后就是随机的了
   liver_slices.sort(key = lambda x: int(x.InstanceNumber))
   # s.pixel_array 获取dicom格式中的像素值
   livers = np.stack([s.pixel_array for s in liver_slices])


   
   image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
   image_slices.sort(key = lambda x: int(x.InstanceNumber))
#================================================
   """ 省略进行的预处理操作，具体见part2"""
   #part2
    # 接part1
   images = get_pixels_hu(image_slices)
   
   images = transform_ctdata(images,500,150)
   
   start,end = getRangImageDepth(livers)
   images = clahe_equalized(images,start,end)
   
   images /= 255.
   # 仅提取腹部所有切片中包含了肝脏的那些切片，其余的不要
  
   total = (end - 4) - (start+4) +1
   print("%d person, total slices %d"%(i,total))
   # 首和尾目标区域都太小，舍弃
   images = images[start+5:end-5]
   print("%d person, images.shape:(%d,)"%(i,images.shape[0]))
   
   livers[livers>0] = 1
   
   livers = livers[start+5:end-5]

   
#=================================================
   full_images.append(images)
   full_livers.append(livers)
   
   full_images = np.vstack(full_images)
   full_images = np.expand_dims(full_images,axis=-1)
   full_livers = np.vstack(full_livers)
   full_livers = np.expand_dims(full_livers,axis=-1)

#=================================================
#part3 接part2
   image_datagen.fit(full_images, augment=True, seed=seed)
   mask_datagen.fit(full_livers, augment=True, seed=seed)
   image_generator = image_datagen.flow(full_images,seed=seed)
   mask_generator = mask_datagen.flow(full_livers,seed=seed)

   train_generator = zip(image_generator, mask_generator)
   x=[]
   y=[]
   i = 0
   for x_batch, y_batch in train_generator:
      i += 1
      x.append(x_batch)
      y.append(y_batch)
      if i>=2: # 因为我不需要太多的数据
         break
   x = np.vstack(x)
   y = np.vstack(y)
#===================================================
#part4 接part3
   dataset.add(full_images, full_livers)
   dataset.add(x, y)
   print('add once finished.')
   # end of lop
dataset.close()

# generate_test.py

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
# from HDF5DatasetWriter import HDF5DatasetWriter
# from HDF5DatasetGenerator import HDF5DatasetGenerator
# from utils import *
from tqdm import tqdm
import cv2

# 可以在part1之前设定好（即循环外）
seed=1
# data_gen_args = dict(rotation_range=3,
#                     width_shift_range=0.01,
#                     height_shift_range=0.01,
#                     shear_range=0.01,
#                     zoom_range=0.01,
#                     fill_mode='nearest')

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()
print('build ImageDataGenerator finished.')

# 可以在part1之前设定好（即循环外）
# 这儿的数量需要提前写好，感觉很不方便，但不知道怎么改，我是先跑了之前的程序，计算了一共有多少
# 张图片后再写的，但这样明显不是好的解决方案
outputPath = "./data_train/val_liver.h5"

if os.path.exists(outputPath):
  os.remove(outputPath)


print('build HDF5DatasetWriter finished')

full_images2 = []
full_livers2 = []
for i in range(18,21):#后3个人作为测试样本
    label_path = '../input/liver20/3Dircadb1/3Dircadb1.%d/MASKS_DICOM/liver'%i
    data_path = '../input/liver20/3Dircadb1/3Dircadb1.%d/PATIENT_DICOM'%i
    liver_slices = [pydicom.dcmread(label_path + '/' + s) for s in os.listdir(label_path)]
    liver_slices.sort(key = lambda x: int(x.InstanceNumber))
    livers = np.stack([s.pixel_array for s in liver_slices])
    start,end = getRangImageDepth(livers)
    total = (end - 4) - (start+4) +1
    print("%d person, total slices %d"%(i,total))
    
    image_slices = [pydicom.dcmread(data_path + '/' + s) for s in os.listdir(data_path)]
    image_slices.sort(key = lambda x: int(x.InstanceNumber))
    
    images = get_pixels_hu(image_slices)
    images = transform_ctdata(images,500,150)
    images = clahe_equalized(images,start,end)
    images /= 255.
    images = images[start+5:end-5]
    print("%d person, images.shape:(%d,)"%(i,images.shape[0]))
    livers[livers>0] = 1
    livers = livers[start+5:end-5]

    full_images2.append(images)
    full_livers2.append(livers)
    
full_images2 = np.vstack(full_images2)
full_images2 = np.expand_dims(full_images2,axis=-1)
full_livers2 = np.vstack(full_livers2)
full_livers2 = np.expand_dims(full_livers2,axis=-1)

dataset = HDF5DatasetWriter(image_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            mask_dims=(full_images2.shape[0], full_images2.shape[1], full_images2.shape[2], 1),
                            outputPath=outputPath)


dataset.add(full_images2, full_livers2)

print("total images in val ",dataset.close())
# partA
import os
import sys
import numpy as np
import random
import math
import tensorflow as tf
# from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage import io

#设置图像的维度顺序（‘tf’或‘th’）
K.set_image_data_format('channels_last')

# partB 接partA
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的validation数据
# part1部分储存的数据文件
outputPath = './data_train/train_liver.h5' # 训练文件
val_outputPath = './data_train/val_liver.h5'
#checkpoint_path = 'model.ckpt'
BATCH_SIZE = 1 # 根据服务器的GPU显存进行调整
 
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        print(target.shape)
        print(refer.shape)
#         cw = (target._keras_shape[2] - refer._keras_shape[2])
        cw = (target.shape[2] - refer.shape[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.shape[1] - refer.shape[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH , 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    
    ch, cw = get_crop_shape(conv4, up_conv5)
    
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    up_conv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)
    
    up7 = concatenate([up_conv6, crop_conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    up_conv7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)

    up8 = concatenate([up_conv7, crop_conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    up_conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)


    up9 = concatenate([up_conv8, crop_conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
# train.py
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
# from HDF5DatasetWriter import HDF5DatasetWriter
# from HDF5DatasetGenerator import HDF5DatasetGenerator
# from utils import *
from tqdm import tqdm
import cv2
from skimage import io
# from Unet import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard



# partB 接partA
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的validation数据
# part1部分储存的数据文件
outputPath = './data_train/train_liver.h5' # 训练文件
val_outputPath = './data_train/val_liver.h5'
#checkpoint_path = 'model.ckpt'
BATCH_SIZE = 8 # 根据服务器的GPU显存进行调整


reader = HDF5DatasetGenerator(dbPath=outputPath,batchSize=BATCH_SIZE)
train_iter = reader.generator()
        
test_reader = HDF5DatasetGenerator(dbPath=val_outputPath,batchSize=BATCH_SIZE)
test_iter = test_reader.generator()
#fixed_test_images, fixed_test_masks = test_iter.__next__()
#   
#def lr_schedule(epoch): return 0.0005 * 0.4**epoch    
model = get_unet()
model_checkpoint = ModelCheckpoint(
        filepath='./models/weights_unet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=False)
# learning_rate = np.array([lr_schedule(i) for i in range(10)])
# reduce_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
tensor_board = TensorBoard(log_dir='./models/logs', write_graph=True)
callbacks = [tensor_board, model_checkpoint, early_stop]#, reduce_lr]
# 注：感觉validation的方式写的不对，应该不是这样弄的
model.fit_generator(train_iter,
                    steps_per_epoch=int(TOTAL/BATCH_SIZE),
                    verbose=1,
                    epochs=500,
                    shuffle=True,
                    validation_data=test_iter,
                    validation_steps=int(TOTAL_VAL/BATCH_SIZE),
                    callbacks=callbacks)
        
reader.close()
test_reader.close()
# test.py
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
# from HDF5DatasetWriter import HDF5DatasetWriter
# from HDF5DatasetGenerator import HDF5DatasetGenerator
# from utils import *
from tqdm import tqdm
import cv2
from skimage import io
# from Unet import *

# partB 接partA
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的validation数据
# part1部分储存的数据文件
outputPath = './data_train/train_liver.h5' # 训练文件
val_outputPath = './data_train/val_liver.h5'
#checkpoint_path = 'model.ckpt'
BATCH_SIZE = 4 # 根据服务器的GPU显存进行调整


print('-'*30)
print('Loading and preprocessing test data...')
test_reader = HDF5DatasetGenerator(dbPath=val_outputPath,batchSize=BATCH_SIZE)
test_iter = test_reader.generator()
fixed_test_images, fixed_test_masks = test_iter.__next__()
print('-'*30)


print('-'*30)
model = get_unet()
print('Loading saved weights...')
print('-'*30)
model.load_weights('./models/dont_change_lr/weights_unet-04--0.85.h5')
    
print('-'*30)
print('Predicting masks on test data...')
imgs_mask_test = model.predict(fixed_test_images, verbose=1)
print('-'*30)



print('-' * 30)
print('Saving predicted masks to files...')
np.save('imgs_mask_test.npy', imgs_mask_test)
print('-' * 30)

pred_dir = 'preds'
if not os.path.exists(pred_dir):
	os.mkdir(pred_dir)

i = 0
for image in imgs_mask_test:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    gt = (fixed_test_masks[i,:,:,0] * 255.).astype(np.uint8)
    ini = (fixed_test_images[i,:,:,0] *255.).astype(np.uint8)
    io.imsave(os.path.join(pred_dir, str(i) + '_ini.png'), ini)
    io.imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
    io.imsave(os.path.join(pred_dir, str(i) + '_gt.png'), gt)
    i += 1

print("total images in test ",str(i))
