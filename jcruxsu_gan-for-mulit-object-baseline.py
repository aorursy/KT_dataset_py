import glob
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers,optimizers
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
noise_size= 100
def DC_GEN():
    gen = Sequential()
    dropout=0.4
    hdim =288
    wdim = 480
    depth = 64 #512
    #2 4 8 16 32 64 128
    gen.add(Dense((hdim*wdim*depth),input_dim=noise_size))
    gen.add(BatchNormalization(momentum=0.9))        
    gen.add(Activation('elu'))
    gen.add(Reshape((hdim,wdim,depth)))
    gen.add(Dropout(dropout))
    gen.add(Conv2DTranspose(int(depth/2),5,padding='same')) #2
    gen.add(BatchNormalization(momentum=0.9))  
    gen.add(Activation('relu'))
    gen.add(Conv2DTranspose(int(depth/4),5,padding='same')) #4
    gen.add(BatchNormalization(momentum=0.9))  
    gen.add(Activation('relu'))
    gen.add(Conv2DTranspose(int(depth/4),5,padding='same')) #8
    gen.add(BatchNormalization(momentum=0.9))  
    gen.add(Activation('relu'))
    gen.add(Conv2DTranspose(int(depth/8),5,padding='same')) #16
    gen.add(BatchNormalization(momentum=0.9))  
    gen.add(Activation('relu'))
    gen.add(Conv2DTranspose(3,5,padding='same')) #32
    gen.add(Activation('sigmoid',name="last"))
    gen.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy')
    return gen
def DC_DIS():
    dis = Sequential()
    input_size = (288, 480, 3)
    depth = 16
    dropout=0.4
    dis.add(Conv2D(depth, 5, padding='same', kernel_initializer='he_normal',input_shape=input_size))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(Dropout(dropout))
    dis.add(Conv2D(depth*2, 5, padding='same', kernel_initializer='he_normal',input_shape=input_size))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(Dropout(dropout))
    dis.add(Conv2D(depth*4, 5, padding='same', kernel_initializer='he_normal',input_shape=input_size))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(Dropout(dropout))  
    dis.add(Conv2D(depth*8, 5, padding='same', kernel_initializer='he_normal',input_shape=input_size))
    dis.add(LeakyReLU(alpha=0.2))
    dis.add(Dropout(dropout)) 
    dis.add(Flatten())
    dis.add(Dense(1))
    dis.add(Activation('sigmoid'))
    dis.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy')
    return dis
dc_gen = DC_GEN()
dc_dis = DC_DIS()
gan = Sequential()
gan.add(dc_gen)
gan.add(dc_dis)
gan.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy')
def train(x_data,batch_size=4):
    images_train = x_data[np.random.randint(0, x_data.shape[0], size=batch_size), :, :, :]
    print(images_train.shape)
    noise = np.random.uniform(-1,1,size=[batch_size,noise_size])
    
    image_fake = dc_gen.predict(noise)
    #plt.imshow(image_fake[0,:,:,0])
    
    x = np.concatenate((images_train,image_fake))
    y = np.ones([2*batch_size,1])
    y[batch_size:,:]=0
    dc_dis.trainable=True
    d_loss = dc_dis.train_on_batch(x,y)
    
    y = np.ones([batch_size,1])
    noise = np.random.uniform(-1,1,size=[batch_size,noise_size])
    dc_dis.trainable=False
    gan_loss = gan.train_on_batch(noise,y)
    return d_loss,gan_loss,image_fake

def save():
    dc_dis.save("dis_for_gan.h5")
    dc_gen.save("dis_for_gan.h5")
trainPointNames=[]
for dirname, _, filenames in os.walk('../input/davis-pointannotation-dataset/Annotations'):
    for filename in filenames:
           trainPointNames.append(os.path.join(dirname, filename))
#Train
trainMask = []
trainImg = []
trainPoint = []
trainPointNames.sort()
for j in range(len(trainPointNames)):
    trainPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png','.jpg')
    trainImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point','/480p')
    trainMask.append(maskpath)
#valid
validMask = []
validImg = []
validPoint = []

for j in range(len(trainPointNames)):
    validPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png', '.jpg')
    validImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point', '/480p')
    validMask.append(maskpath)
train_x = np.zeros((len(trainImg),288,480, 3))
valid_x = np.zeros((len(validImg),288,480, 3))
train_mask_y = np.zeros((len(trainMask),288,480))
valid_mask_y = np.zeros((len(validMask),288,480))
#train
for index in range(len(trainImg)):
    mask = cv2.imread(trainMask[index], cv2.IMREAD_GRAYSCALE)
    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    img = cv2.imread(trainImg[index], cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    #Point = cv2.imread(trainPoint[index], cv2.IMREAD_GRAYSCALE)
    #dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    tmask = np.array(dstmask)
    timg = np.array(dstimg)
    #train_Point = np.array(dstPoint)
    train_x[index,:,:,0:3] = timg
    #train_x[index,:,:,3] = train_Point
    train_mask_y[index,:,:] = tmask
#valid
for index in range(len(validImg)):
    mask = cv2.imread(validMask[index], cv2.IMREAD_GRAYSCALE)
    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    img = cv2.imread(validImg[index], cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    #Point = cv2.imread(validPoint[index], cv2.IMREAD_GRAYSCALE)
    #dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vmask = np.array(dstmask)
    vimg = np.array(dstimg)
    #valid_Point =np.array(dstPoint)
    valid_x[index, :, :, 0:3] = vimg
    #valid_x[index, :, :, 3] = valid_Point
    valid_mask_y[index,:,:] = vmask

epochs=1
sample_size=10
batch_size=4
train_for_epoch = train_x.shape[0]//batch_size
for epoch in range(0,epochs):
    total_d_loss=0
    total_g_loss=0
    imgs = None
    
    for batch in range(0,train_for_epoch):
        d_loss,a_loss,t_imgs = train(train_x,batch_size)
        total_d_loss +=d_loss
        total_g_loss +=a_loss
        if img is None:
            imgs = t_imgs
    if epoch%2 ==0 or epoch==epochs:
        total_d_loss /=train_for_epoch
        total_g_loss /=train_for_epoch
        print("Epoch :{}, D Loss : {}, G LOSS : {}".format(epoch,total_d_loss,total_g_loss))
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1)) 
        for i in range(0, sample_size): 
            ax[i].set_axis_off() 
            ax[i].imshow(imgs[i].reshape((gan.img_rows, gan.img_cols, gan.channel)), interpolation='nearest'); 
            plt.show() 
        save()

    
