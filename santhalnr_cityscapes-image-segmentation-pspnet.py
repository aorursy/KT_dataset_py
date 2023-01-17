import numpy as np

import time

import os

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from tensorflow import keras

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Convolution2D,BatchNormalization,ReLU,LeakyReLU,Add,Activation

from tensorflow.keras.layers import GlobalAveragePooling2D,AveragePooling2D,UpSampling2D
train_folder="/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/train/"

valid_folder="/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/val/"



def get_images_masks(path):

    names=os.listdir(path)

    img_g,img_m=[],[]

    for name in names:

        img=cv2.imread(path+name)

        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)

        img=img[:,:,::-1]

        img_g.append(img[:,:256])

        img_m.append(np.reshape(img[:,256:],(256*256*3)))

        del img

    del names

    return img_g,img_m

        

train_imgs,train_masks=get_images_masks(train_folder)

valid_imgs,valid_masks=get_images_masks(valid_folder)



#train_len=len(train_imgs)

#valid_len=len(valid_imgs)

#print(f'Train Images:{train_len}\nValid Images:{valid_len}')
def conv_block(X,filters,block):

    # resiudal block with dilated convolutions

    # add skip connection at last after doing convoluion operation to input X

    

    b = 'block_'+str(block)+'_'

    f1,f2,f3 = filters

    X_skip = X

    # block_a

    X = Convolution2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),

                      padding='same',kernel_initializer='he_normal',name=b+'a')(X)

    X = BatchNormalization(name=b+'batch_norm_a')(X)

    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)

    # block_b

    X = Convolution2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),

                      padding='same',kernel_initializer='he_normal',name=b+'b')(X)

    X = BatchNormalization(name=b+'batch_norm_b')(X)

    X = LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)

    # block_c

    X = Convolution2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),

                      padding='same',kernel_initializer='he_normal',name=b+'c')(X)

    X = BatchNormalization(name=b+'batch_norm_c')(X)

    # skip_conv

    X_skip = Convolution2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)

    X_skip = BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)

    # block_c + skip_conv

    X = Add(name=b+'add')([X,X_skip])

    X = ReLU(name=b+'relu')(X)

    return X

    

def base_feature_maps(input_layer):

    # base covolution module to get input image feature maps 

    

    # block_1

    base = conv_block(input_layer,[32,32,64],'1')

    # block_2

    base = conv_block(base,[64,64,128],'2')

    # block_3

    base = conv_block(base,[128,128,256],'3')

    return base



def pyramid_feature_maps(input_layer):

    # pyramid pooling module

    

    base = base_feature_maps(input_layer)

    # red

    red = GlobalAveragePooling2D(name='red_pool')(base)

    red = tf.keras.layers.Reshape((1,1,256))(red)

    red = Convolution2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(red)

    red = UpSampling2D(size=256,interpolation='bilinear',name='red_upsampling')(red)

    # yellow

    yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)

    yellow = Convolution2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(yellow)

    yellow = UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)

    # blue

    blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)

    blue = Convolution2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)

    blue = UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)

    # green

    green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)

    green = Convolution2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(green)

    green = UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)

    # base + red + yellow + blue + green

    return tf.keras.layers.concatenate([base,red,yellow,blue,green])



def last_conv_module(input_layer):

    X = pyramid_feature_maps(input_layer)

    X = Convolution2D(filters=3,kernel_size=3,padding='same',name='last_conv_3_by_3')(X)

    X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)

    X = Activation('sigmoid',name='last_conv_relu')(X)

    X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)

    return X
input_layer = tf.keras.Input(shape=np.squeeze(train_imgs[0]).shape,name='input')

output_layer = last_conv_module(input_layer)

model = tf.keras.Model(inputs=input_layer,outputs=output_layer)
model.summary()
model.load_weights('/kaggle/input/best-modelh5/best_model.h5')
'''

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')

model.fit(np.array(train_imgs,dtype='float16'),np.array(train_masks,dtype='float16'),

          validation_data=(np.array(valid_imgs,dtype='float16'),np.array(valid_masks,dtype='float16')),

          epochs=20,steps_per_epoch=297,verbose=1,batch_size=10)

'''

#tf.keras.models.save_model(model,'/kaggle/working/best_model.h5')
def plot_imgs(img,mask,pred):

    mask = np.reshape(mask,(256,256,3))

    pred = np.reshape(pred,(256,256,3))

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,10))

    ax1.imshow(img)

    ax1.axis('off')

    ax2.imshow(mask)

    ax2.axis('off')

    ax3.imshow(pred)

    ax3.axis('off')
pred_masks = model.predict(np.array(valid_imgs,dtype='float16'))
print('-------------Input---------------Actual mask--------------Predicted mask-------')

for i in range(5):

    x = np.random.randint(0,500,size=1)[0]

    plot_imgs(valid_imgs[x],valid_masks[x],pred_masks[x])