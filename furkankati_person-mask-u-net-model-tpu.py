from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path("person-masking-dataset")
print(GCS_DS_PATH)
from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras

img_rows = 512
img_cols = 512
testBatchSize = 1
smooth = 1.
Epochs = 10
fileCount = 8000

basePath = '/kaggle/input/person-masking-dataset/'

inputDir = basePath + "unetinput/"
outputDir = basePath + "unetoutput/"

testDatPath = basePath + "test3/"
pred_dir = basePath + "preds/"

weightsName = "weightGeneratorTrainPersonMask.h5"
#weightsName = "weightGeneratorAccuracy.h5"
file_list = []
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
print("Done")

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.
batchSize = 4*tpu_strategy.num_replicas_in_sync
print(batchSize)
def get_unet():
    inputs = Input((img_rows,img_cols,1))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


def generate_test_data(directory, batch_size):
    i = 0
    file_list = os.listdir(directory)
  
    count = 0
    while True:
        image_batch_person = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            count += 1    
            sample = file_list[i]
            i += 1
            fileName = directory +sample
            print("test :" + fileName )
            image = cv2.imread(fileName,0)
            image = cv2.resize(image,(img_cols,img_rows),cv2.INTER_AREA)
            image = np.expand_dims(image,2)
            image_batch_person.append(image)
                    
        image_batch_person      = np.asarray(image_batch_person,np.float64)
         
        image_batch_person /= 255.
        yield image_batch_person
def generate_data(batch_size,file_listParm):
    i = 0
    
    while True:
        image_batch_person = []
        image_batch_mask = []
        for b in range(batch_size):
            if i == len(file_listParm):
                i = 0
                random.shuffle(file_listParm)
            sample = file_listParm[i]
            i += 1
            #print(str(i))
            inputFileName   = inputDir +sample
            outputFileName  = outputDir + sample
            
            inputImage      = cv2.imread(inputFileName)
            outputImage     = cv2.imread(outputFileName)
            
            #print(inputFileName)
            #print(outputFileName)
            
            if(type(inputImage) !=type(None) and type(outputImage) !=type(None) ):
                if(inputImage.size  != 0 and outputImage.size != 0):
                    inputImage      = cv2.resize(inputImage,(img_rows,img_cols),cv2.INTER_AREA)
                    outputImage     = cv2.resize(outputImage,(img_rows,img_cols),cv2.INTER_AREA)
                    
                    inputImage      = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
                    outputImage     = cv2.cvtColor(outputImage,cv2.COLOR_BGR2GRAY)
                    
                    inputImage      = np.expand_dims(inputImage,2)
                    outputImage     = np.expand_dims(outputImage,2)
            
                    image_batch_person.append(inputImage)
                    image_batch_mask.append(outputImage)
                    
        image_batch_person      = np.asarray(image_batch_person,np.float64)
        image_batch_mask        = np.asarray(image_batch_mask,np.float64)
           
        image_batch_mask /= 255.           
        image_batch_person /= 255. 
        
        yield image_batch_person,image_batch_mask
        
def load_and_predict(model):
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(weightsName)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    
    totalTestData = len(os.listdir(testDatPath))
    print("total test data :" + str(totalTestData)   )
    test_generator =  generate_test_data(testDatPath,testBatchSize )
    predict = model.predict_generator(test_generator,steps = totalTestData//testBatchSize )
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    
    
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    imgtestID = 0    
    for image in predict:
        image = (image[:, :, 0] * 255.).astype(np.uint8) 
        filenameSave = os.path.join(pred_dir, str(imgtestID) + '_pred.jpg')
        print(filenameSave)
        cv2.imwrite(filenameSave, image)
        imgtestID += 1
        
def build_load_and_predict():
    file_list = os.listdir(inputDir)
    model = get_unet()
    load_and_predict(model)
    
def train_and_predict():
    file_list = os.listdir(inputDir)
    #file_list = file_list[:fileCount]
    
    with strategy.scope():
        model = get_unet()
    
    totalData = len(file_list)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    datagenerator = generate_data(batchSize,file_list)
    model.fit_generator(generator=datagenerator,
                        epochs=Epochs,
                        steps_per_epoch=totalData // batchSize)
    
    model.save_weights(weightsName)
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    
    load_and_predict(model)


if __name__ == '__main__':
    train_and_predict()
    #build_load_and_predict()
    print('done')
