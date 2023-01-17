import os
import re

import matplotlib.pyplot as plt
from scipy import ndimage, misc
from matplotlib import pyplot
import cv2 as cv
import numpy as np
np.random.seed(0)

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
print(tf.__version__)
print(ndimage.__version__)
input_img = Input(shape=(256, 256, 3))
l1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l4 = Dropout(0.3)(l3)
l5 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l5)
l7 = MaxPooling2D(padding='same')(l6)
l8 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l7)
l9 = UpSampling2D()(l8)
l10 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)
l11 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l10)
l12 = add([l6,l11])
l13 = UpSampling2D()(l12)
l14 = Conv2D(64, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)
l15 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l14)
l16 = add([l15,l2])
decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l16)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
from tensorflow.keras.optimizers import Adam
def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam
optimizer=get_optimizer()
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['acc'])
TRAIN_HR_DIR='../input/cars-train-img/cars_train'
import os
import re
from scipy import ndimage, misc
from skimage import data, color
from skimage.transform import resize, rescale
from matplotlib import pyplot
import numpy as np
import gc

broken_file=[]
epoch=1
scale_percent = 50
for epoch in range(epoch):
    # 1 is only for demo I use 10 for my training
    for i in range(1):
        x_train_n2=[]
        x_train_down2=[]
        x_train_n=[]
        x_train_down=[]
        for a in range(800):
            broken_file=[]
             #a=a+301 #loop 301-350
              #a=a+401 #loop(401-600)
            counter=(i+7)*800
            a=a+1+counter 
            b = (lambda a: '{0:0>5}'.format(a))(a)
                
            filename = str(b)+'.jpg'
            #print(filename)
            image=0
            image_resized=0
            
            image_path=os.path.join(TRAIN_HR_DIR,filename)
            image = pyplot.imread(image_path)
        
            image_resized = resize(image, (256,256))
            broken_file.append(image_resized)
            dim_test=np.array(broken_file)
            if(dim_test.ndim==4):
                x_train_n.append(image_resized)
            
                low_res_image=resize(resize(image, (128, 128)),(256,256))
            
                x_train_down.append(low_res_image)
                            
                
                
                x_train_n2=np.array(x_train_n).reshape(-1, 256,256, 3)
                
                x_train_down2=np.array(x_train_down).reshape(-1, 256,256, 3)
            else:
                error="file error in file no."+filename
                print(error)
                
        hist=autoencoder.fit(x_train_down2, x_train_n2,
                        epochs=2,
                        batch_size=10,
                        shuffle=True,
                        validation_split=0.15)
        del x_train_n2
        del x_train_down2
        del x_train_n
        del x_train_down
        # Somehow when I add gc.collect() it even saves more spaces for training
        gc.collect()
autoencoder.save('car_sr_8000.hdf5') 

def get_dataset():
    broken_file=[]
    epoch=1
    scale_percent = 50
    x_train_n=[]
    x_train_down=[]
    x_train_n2=[]
    x_train_down2=[]
    for a in range(100):
        broken_file=[]
        #a=a+301 #loop 301-350
        #a=a+401 #loop(401-600)
        
        a=a+1
        b = (lambda a: '{0:0>5}'.format(a))(a)
        filename = str(b)+'.jpg'
        #print(filename)
        image=0
        image_resized=0
        
        image_path=os.path.join(TRAIN_HR_DIR,filename)
        image = pyplot.imread(image_path)
        
        image_resized = resize(image, (256,256))
        broken_file.append(image_resized)
        dim_test=np.array(broken_file)
        if(dim_test.ndim==4):
            x_train_n.append(image_resized)
            
            low_res_image=resize(resize(image, (128, 128)),(256,256))
            # rescale somehow didn't work in Kaggle it transform my (256,256,3)to(256,256,4)
            #x_train_down.append(rescale(rescale(image_resized, 0.5), 2.0))
            x_train_down.append(low_res_image)
                    
                
                
            x_train_n2=np.array(x_train_n).reshape(-1, 256,256, 3)
            #print(x_train_n2.shape)
            x_train_down2=np.array(x_train_down).reshape(-1, 256,256, 3)
        else:
            error="file error in file no."+filename
            print(error)
        
    return x_train_n2,x_train_down2

x_train_n,x_train_down = get_dataset()
encoded_imgs = autoencoder.predict(x_train_down)
image_index = np.random.randint(0,100)
plt.figure(figsize=(128, 128))
i = 1
ax = plt.subplot(10, 10, i)
# The model low_resolution input
plt.imshow(x_train_down[image_index])


#ax = plt.subplot(10, 10, i)
#plt.imshow(encoded_imgs[image_index].reshape((64*64, 256)))
i += 1
ax = plt.subplot(10, 10, i)
# My model prediction
plt.imshow(encoded_imgs[image_index])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n[image_index])
plt.show()
# The Zoomed vesion of these images can be accessed here
# https://drive.google.com/drive/folders/1ZLMjXPxqQf8vzwD6F23eJM3Yi7g3ImCf?usp=sharing
# picture name : 1st_prediction.png
# We can see that my model sharpends the blurry images
# Note that my 1st model are trained from images that are processed by scikit transform library

# Now Let me pre-load my 2nd model that are trained from images that are processed by cv2 rescale library
input_img = Input(shape=(256, 256, 3))
l1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l4 = Dropout(0.3)(l3)
l5 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l5)
l7 = MaxPooling2D(padding='same')(l6)
l8 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l7)

l9 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(l8)

l10 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)
l11 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l10)
l12 = add([l6,l11])

l13 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(l12)

l14 = Conv2D(64, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)
l15 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l14)
l16 = add([l15,l2])
decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l16)

autoencoder2 = Model(input_img, decoded)
autoencoder2.summary()
autoencoder2.compile(optimizer=optimizer, loss='mean_squared_error')
def get_dataset_2():
    # Using Cv2 for creating pixelated images
    broken_file=[]
    epoch=1
    scale_percent = 25
    x_train_n=[]
    x_train_down=[]
    x_train_n2=[]
    x_train_down2=[]
    for a in range(100):
        broken_file=[]
        #a=a+301 #loop 301-350
        #a=a+401 #loop(401-600)
        
        a=a+1
        b = (lambda a: '{0:0>5}'.format(a))(a)
        filename = str(b)+'.jpg'
        #print(filename)
        image=0
        image_resized=0
        
        image_path=os.path.join(TRAIN_HR_DIR,filename)
        image = pyplot.imread(image_path)
        
        image_resized = resize(image, (256,256))
        broken_file.append(image_resized)
        dim_test=np.array(broken_file)
        if(dim_test.ndim==4):
            x_train_n.append(image_resized)
            
            width = int(image_resized.shape[1] * scale_percent / 100)
            height = int(image_resized.shape[0] * scale_percent / 100)
            dim = (width, height)
            small_image = cv.resize(image_resized, dim, interpolation = cv.INTER_AREA)

            # scale back to original size
            width = int(small_image.shape[1] * 100 / scale_percent)
            height = int(small_image.shape[0] * 100 / scale_percent)
            dim = (width, height)
            low_res_image = cv.resize(small_image, dim, interpolation =  cv.INTER_AREA)
            
            x_train_down.append(low_res_image)
                    
                
                
            x_train_n2=np.array(x_train_n).reshape(-1, 256,256, 3)
            #print(x_train_n2.shape)
            x_train_down2=np.array(x_train_down).reshape(-1, 256,256, 3)
        else:
            error="file error in file no."+filename
            print(error)
        
    return x_train_n2,x_train_down2

x_train_n_cv,x_train_down_cv = get_dataset_2()
autoencoder2.load_weights('../input/my-other-pretrained-model/ResNet_Super_Resolution_7epoch_900data_convTranspose.hdf5')
SR_cv = autoencoder2.predict(x_train_down_cv)
# Trying to see the performance differences
SR = autoencoder.predict(x_train_down_cv)
plt.figure(figsize=(128, 128))
i = 1
ax = plt.subplot(10, 10, i)
# The model low_resolution input
plt.imshow(x_train_down_cv[39])


#ax = plt.subplot(10, 10, i)
#plt.imshow(encoded_imgs[image_index].reshape((64*64, 256)))
i += 1
ax = plt.subplot(10, 10, i)
# My 2nd model prediction (trained on cv2 processed images)
plt.imshow(SR_cv[39])
i += 1
ax = plt.subplot(10, 10, i)
# My 1st model prediction (trained on scikit transform images)
plt.imshow(SR[39])
i += 1
ax = plt.subplot(10, 10, i)
plt.imshow(x_train_n_cv[39])
plt.show()
# The Zoomed vesion of these images can be accessed here
# https://drive.google.com/drive/folders/1ZLMjXPxqQf8vzwD6F23eJM3Yi7g3ImCf?usp=sharing
# picture name : 2nd_prediction.png
# My 1st model works well for enhancing blurry images (prepared by skitransform tools)
# While my 2nd model outperforms my 1st model for enhancing pixelated images (prepared by CV2)
# Super Resolution models performs differently based on how we prepare their training data.

# Notes to self : try to combine dataset that are processed both from cv2 and scikit transform
# Changing loss function to perceptional loss function
# Use dynamic regularization and Callbacks(maybe?)