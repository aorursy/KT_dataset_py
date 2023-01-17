from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import keras
from keras import backend as backendKeras 
import time
from keras.models import load_model
import sys
learningRate = 0.0001
img_rows = 512
img_cols = 512
batchSize = 8
Epochs = 1
#fileCount = 85000

basePath = '/kaggle/input/person-segmentation-dataset/'

inputDir = basePath+"Training/input/"
outputDir = basePath+"Training/Output/"
pretrainedModel = basePath + "UnetColor_pc_195000_1.hdf5"

file_list = []
class MySequence(keras.utils.Sequence):
  def __init__(self, num_batches,image_filenames,batch_size):
      self.num_batches = num_batches
      self.image_filenames = image_filenames
      self.batch_size = batch_size
      
  def __len__(self):
      return self.num_batches # the length is the number of batches

  def __getitem__(self, idx):
     halfSize = self.batch_size//2
     start = idx * halfSize  
        
     global inputDir
     global outputDir
     
     image_batch_person = np.ndarray((batchSize, img_cols,img_rows, 3), dtype=np.float64)
     image_batch_mask = np.ndarray((batchSize, img_cols,img_rows, 1), dtype=np.float64)
     
     imageIndex = 0
     for imageIndex in range(halfSize):
         trainImageFileName = self.image_filenames[start+imageIndex]
         inputFilePath      = inputDir + trainImageFileName
         outputFilePath     = outputDir + trainImageFileName.replace("jpg","png",1)
         
         inputImage         = cv2.imread(inputFilePath)
         outputImage        = cv2.imread(outputFilePath,0) #çıktı resmi gri bir resimdir
         
         flippedOutputImage = cv2.flip(outputImage,1)
         flippedInputImage  = cv2.flip(inputImage,1)
        
         outputImage            = np.expand_dims(outputImage,2)#gri resim olduğu içindir
         flippedOutputImage     = np.expand_dims(flippedOutputImage,2)#gri resim olduğu içindir
        
         image_batch_person[imageIndex] = inputImage
         image_batch_mask[imageIndex] = outputImage
         
         newIndex = imageIndex+halfSize
         image_batch_person[newIndex] = flippedInputImage
         image_batch_mask[newIndex] = flippedOutputImage
        
     image_batch_person = image_batch_person/255
     image_batch_mask = image_batch_mask/255        
            
     return image_batch_person,image_batch_mask

def get_unet(DropoutRate = 0.5):
    inputs = Input((img_rows, img_cols, 3))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(DropoutRate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(DropoutRate)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9],  axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def buildPretrainedModel():
    model = load_model(pretrainedModel)
    
    return model
 
def train_on_batch():
    file_list = os.listdir(inputDir)
    fileCount = len(file_list)
    model = buildPretrainedModel()
    #model = get_unet(DropoutRate=0.5)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    steps_per_epoch = fileCount // (batchSize)
    
    MySequenceGenerator = MySequence(steps_per_epoch ,file_list,batchSize)
    
    
    model.fit_generator(generator=MySequenceGenerator,
	                   steps_per_epoch = steps_per_epoch,
	                   epochs = Epochs,
	                   verbose = 1,
	                   workers = 8)
    
    weightsName = "UnetColor_kaggle_" + str(fileCount) + "_" + str(Epochs) + ".h5"
    checkpointPath = "UnetColor_kaggle_"+ str(fileCount) + "_" + str(Epochs) + ".hdf5"
    
    model.save_weights(weightsName)  
    model.save(checkpointPath)
    
if __name__ == '__main__':
    start_time = time.time()
    
    train_on_batch()
    print('done')