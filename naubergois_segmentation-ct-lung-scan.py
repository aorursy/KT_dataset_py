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

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from nibabel.testing import data_path

import os
example_filename = os.path.join(data_path, '/kaggle/input/covid19-ct-scans/ct_scans/coronacases_org_001.nii')
example_filename 
import matplotlib.pyplot as plt



def multi_slice_viewer(volume):

    #remove_keymap_conflicts({'j', 'k'})

    fig, ax = plt.subplots()

    ax.volume = volume

    ax.index = volume.shape[0] // 2

    ax.imshow(volume[ax.index])

    fig.canvas.mpl_connect('key_press_event', process_key)



def process_key(event):

    fig = event.canvas.figure

    ax = fig.axes[0]

    if event.key == 'j':

        previous_slice(ax)

    elif event.key == 'k':

        next_slice(ax)

    fig.canvas.draw()



def previous_slice(ax):

    volume = ax.volume

    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %

    ax.images[0].set_array(volume[ax.index])



def next_slice(ax):

    volume = ax.volume

    ax.index = (ax.index + 1) % volume.shape[0]

    ax.images[0].set_array(volume[ax.index])
import nibabel as nib
img = nib.load(example_filename)
img.shape


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(20):

    plt.subplot(5, 5, i + 1)



    plt.imshow(im_fdata[:,:,i])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(20):

    plt.subplot(5, 5, i + 1)



    plt.imshow(im_fdata[i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  
def show_images(images):



    n_ = min(images.shape[0], 20) 

    rows = 4

    cols = (n_ // 4) + (1 if (n_ % 4) != 0 else 0)

    figure = plt.figure(figsize=(2*rows, 2*cols))

    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)

    for i in range(n_):

        plt.subplot(cols, rows, i + 1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        if images.shape[1] == 3:

           

            vol = images[i].detach().numpy()

            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \

                            for y in range(vol.shape[2])] \

                            for x in range(vol.shape[1])]

            plt.imshow(img)

        else: 

            plt.imshow((images[i, 0]*255).int(), cmap= "gray")



    return figure
!pip install medpy

import os

from os import listdir

from os.path import isfile, join

from medpy.io import load

import cv2

import numpy as np



def LoadLungData(x_shape, y_shape,limit):





    image_dir = '/kaggle/input/covid19-ct-scans/ct_scans/'

    label_dir = '/kaggle/input/covid19-ct-scans/lung_and_infection_mask/'



    images = [f for f in listdir(image_dir) if (

        isfile(join(image_dir, f)) and f[0] != ".")]



    out = []

    count=0

    for f in images:

        

        count+=1

        if count<limit:

            #print('Count ',count)

            #print('Limit ',limit)



            image, _ = load(os.path.join(image_dir, f))

            label, _ = load(os.path.join(label_dir, f.replace('org_','')

                                         .replace('org_covid-19-pneumonia-','')

                                         .replace('covid-19-pneumonia-','')

                                        .replace('-dcm','')))



            if image.shape[0]!=label.shape[0]:

               print('File and label with different shapes ',f)





            #image=image/255



            try:



                image = reshape(image, new_shape=( x_shape, y_shape,576))

                #print('Image shape ',image.shape)

                label = reshape(label, new_shape=( x_shape, y_shape,576)).astype(int)

            except:

                print('Error in file ',f)

                raise



            out.append({"image": image, "seg": label, "filename": f})

        else:

            break



 

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
def reshape(image, new_shape):

  

    reshaped_image = np.zeros(new_shape)



    print(reshaped_image.shape)

    print(image.shape)

    range_0=reshaped_image.shape[0]-image.shape[0]

    range_1=reshaped_image.shape[1]-image.shape[1]

    range_2=reshaped_image.shape[2]-image.shape[2]

    

    # if ((range_0>=0) and (range_1>=0)) and  (range_2>=0):

    #reshaped_image[0:image.shape[0],0:image.shape[1],0:image.shape[2]]+=image

    for i in range(image.shape[2]):

        

        reshaped_image[:,:,i]=cv2.resize(image[:,:,i], ( 64, 64))

    reshaped_image=reshaped_image.transpose(2,1,0)

    print('Image reshaped ',reshaped_image.shape)

    # else:

    #   raise Exception("Invalid file shape")

       

        

    return reshaped_image
# out=LoadLungData(64, 64,100)

# print(out.shape)
# out.shape


# import matplotlib.pyplot as plt

# im_fdata=img.get_fdata()









# plt.figure(figsize=(10,8))



# # Iterate and plot random images

# for i in range(20):

#     plt.subplot(5, 5, i + 1)



#     plt.imshow(out[0]['image'][i,:,:])

#     plt.axis('off')

    

# # Adjust subplot parameters to give specified padding

# plt.tight_layout()  


# import matplotlib.pyplot as plt

# im_fdata=img.get_fdata()









# plt.figure(figsize=(10,8))



# # Iterate and plot random images

# for i in range(20):

#     plt.subplot(5, 5, i + 1)



#     plt.imshow(out[0]['seg'][i,:,:])

#     value=out[0]['seg'][i,:,:]>0

#     print(value)

#     plt.axis('off')

    

# # Adjust subplot parameters to give specified padding

# plt.tight_layout()  


# import matplotlib.pyplot as plt

# im_fdata=img.get_fdata()









# plt.figure(figsize=(10,8))



# # Iterate and plot random images

# for i in range(20):

#     plt.subplot(5, 5, i + 1)



#     plt.imshow(out[0]['image'][:,:,i])

#     plt.axis('off')

    

# # Adjust subplot parameters to give specified padding

# plt.tight_layout()  


# import matplotlib.pyplot as plt

# im_fdata=img.get_fdata()









# plt.figure(figsize=(10,8))



# # Iterate and plot random images

# for i in range(10,35):

#     plt.subplot(5, 5, i + 1-10)



#     plt.imshow(out[0]['seg'][:,:,i])

#     plt.axis('off')

    

# # Adjust subplot parameters to give specified padding

# plt.tight_layout()  
# keys = range(len(out))

# split = dict()

# size=len(out)

# split_1=int(size*0.7)

# split_2=int(size*0.7)+int(size*0.2)

# split['train']=range(0,split_1)

# split['test']=range(split_1,split_2)

# split['val']=range(split_2,size)

# print('len val',len(split['val']))
import os

import time



import numpy as np

import torch

import torch.optim as optim

import torch.nn.functional as F



from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
n_epochs =10

time_start = ""

time_end = ""

epoch = 0

 
# import torch

# from torch.utils.data import Dataset





# class SlicesDataset(Dataset):



#     def __init__(self, data):

#         self.data = data



#         self.slices = []



#         for i, d in enumerate(data):

#             print(d["image"].shape[0])

#             for j in range(d["image"].shape[0]):

#                 self.slices.append((i, j))

#         print('Len slices ',len(self.slices))



#     def __getitem__(self, idx):



#         slc = self.slices[idx]

#         sample = dict()

#         sample["id"] = idx





#         i,j=slc

        

#         #print('i ',i)

#         #print('j ',j)

    

#         import numpy as np



#         image_=self.data[i]['image']

#         label_=self.data[i]['seg']

#         image=image_[j,:,:]

#         #print('Slice shape ',image.shape)

#         print('1',image.shape)

#         image=image.reshape(1,image.shape[0],image.shape[1])

#         print('2',image.shape)

#         label=label_[j,:,:]

#         label=label.reshape(1,label.shape[0],label.shape[1])

   

#         sample['image']=torch.tensor(image)#

#         sample['seg']=torch.tensor(label)#



#         return sample



#     def __len__(self):

   

#         return len(self.slices)
import numpy as np



import matplotlib

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from nilearn.surface import surface

from nilearn.plotting import show
# train_loader = DataLoader(SlicesDataset(out[split["train"]]),

#                 batch_size=20, shuffle=True, num_workers=0)

# val_loader = DataLoader(SlicesDataset(out[split["val"]]),

#                 batch_size=20, shuffle=True, num_workers=0)
# test_data = out[split["test"]]
def build_model(inp_shape, k_size=3):

    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)

    data = Input(shape=inp_shape)

    conv1 = Convolution3D(padding='same', filters=32, kernel_size=k_size)(data)

    conv1 = BatchNormalization()(conv1)

    conv1 = Activation('relu')(conv1)

    conv2 = Convolution3D(padding='same', filters=32, kernel_size=k_size)(conv1)

    conv2 = BatchNormalization()(conv2)

    conv2 = Activation('relu')(conv2)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)



    conv3 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(pool1)

    conv3 = BatchNormalization()(conv3)

    conv3 = Activation('relu')(conv3)

    conv4 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv3)

    conv4 = BatchNormalization()(conv4)

    conv4 = Activation('relu')(conv4)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)



    conv5 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(pool2)

    conv5 = BatchNormalization()(conv5)

    conv5 = Activation('relu')(conv5)

    conv6 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv5)

    conv6 = BatchNormalization()(conv6)

    conv6 = Activation('relu')(conv6)

    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv6)



    conv7 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool3)

    conv7 = BatchNormalization()(conv7)

    conv7 = Activation('relu')(conv7)

    conv8 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv7)

    conv8 = BatchNormalization()(conv8)

    conv8 = Activation('relu')(conv8)

    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv8)



    conv9 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(pool4)

    conv9 = BatchNormalization()(conv9)

    conv9 = Activation('relu')(conv9)



    up1 = UpSampling3D(size=(2, 2, 2))(conv9)

    conv10 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(up1)

    conv10 = BatchNormalization()(conv10)

    conv10 = Activation('relu')(conv10)

    conv11 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(conv10)

    conv11 = BatchNormalization()(conv11)

    conv11 = Activation('relu')(conv11)

    merged1 = concatenate([conv11, conv8], axis=merge_axis)

    conv12 = Convolution3D(padding='same', filters=128, kernel_size=k_size)(merged1)

    conv12 = BatchNormalization()(conv12)

    conv12 = Activation('relu')(conv12)



    up2 = UpSampling3D(size=(2, 2, 2))(conv12)

    conv13 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up2)

    conv13 = BatchNormalization()(conv13)

    conv13 = Activation('relu')(conv13)

    conv14 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv13)

    conv14 = BatchNormalization()(conv14)

    conv14 = Activation('relu')(conv14)

    merged2 = concatenate([conv14, conv6], axis=merge_axis)

    conv15 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged2)

    conv15 = BatchNormalization()(conv15)

    conv15 = Activation('relu')(conv15)



    up3 = UpSampling3D(size=(2, 2, 2))(conv15)

    conv16 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up3)

    conv16 = BatchNormalization()(conv16)

    conv16 = Activation('relu')(conv16)

    conv17 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv16)

    conv17 = BatchNormalization()(conv17)

    conv17 = Activation('relu')(conv17)

    merged3 = concatenate([conv17, conv4], axis=merge_axis)

    conv18 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged3)

    conv18 = BatchNormalization()(conv18)

    conv18 = Activation('relu')(conv18)



    up4 = UpSampling3D(size=(2, 2, 2))(conv18)

    conv19 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(up4)

    conv19 = BatchNormalization()(conv19)

    conv19 = Activation('relu')(conv19)

    conv20 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv19)

    conv20 = BatchNormalization()(conv20)

    conv20 = Activation('relu')(conv20)

    merged4 = concatenate([conv20, conv2], axis=merge_axis)

    conv21 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(merged4)

    conv21 = BatchNormalization()(conv21)

    conv21 = Activation('relu')(conv21)



    conv22 = Convolution3D(padding='same', filters=2, kernel_size=k_size)(conv21)

    output = Reshape([-1, 2])(conv22)

    output = Activation('softmax')(output)

    output = Reshape(inp_shape[:-1] + (2,))(output)



    model = Model(data, output)

    return model
# out[0]['image'].shape
# len(out)
# # X=[out[i]['image'] for i in range(len(out))]

# y=[out[i]['seg'] for i in range(len(out))]
# X=np.array(X)

# # y=np.array(y)
# X.shape
# X.shape
# y.shape
# X.shape[1:]
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D

import tensorflow as tf



def Unet3D(inputs,num_classes):

    x=inputs

    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same',data_format="channels_last")(x)

    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv1)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(pool1)

    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(pool2)

    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv3)

    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool3)

    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)



    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool4)

    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv5)

    drop5 = Dropout(0.5)(conv5)



    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))

    merge6 = concatenate([drop4,up6],axis=-1)

    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge6)

    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv6)



    up7 = Conv3D(32, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))

    merge7 = concatenate([conv3,up7],axis=-1)

    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(merge7)

    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv7)



    up8 = Conv3D(16, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))

    merge8 = concatenate([conv2,up8],axis=-1)

    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(merge8)

    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv8)



    up9 = Conv3D(8, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))

    merge9 = concatenate([conv1,up9],axis=-1)

    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(merge9)

    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv3D(1,1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inputs, outputs = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=1.):



    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)

    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)



    intersection = tf.reduce_sum(y_true_f * y_pred_f)



    if loss_type == 'jaccard':

        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))



    elif loss_type == 'sorensen':

        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)



    else:

        raise ValueError("Unknown `loss_type`: %s" % loss_type)



    return (1-(2. * intersection + smooth) / (union + smooth))
def dice_coe(y_true,y_pred, loss_type='jaccard', smooth=1.):



    y_true_f = tf.reshape(y_true,[-1])

    y_pred_f = tf.reshape(y_pred,[-1])



    intersection = tf.reduce_sum(y_true_f * y_pred_f)



    if loss_type == 'jaccard':

        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))



    elif loss_type == 'sorensen':

        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)



    else:

        raise ValueError("Unknown `loss_type`: %s" % loss_type)



    return (2. * intersection + smooth) / (union + smooth)
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# initial_epoch_of_training=0

# TRAIN_CLASSIFY_LEARNING_RATE =1e-4

# OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)



# INPUT_PATCH_SIZE=(576,64,64, 1)

# #with tpu_strategy.scope():

# inputs = tf.keras.Input(shape=(INPUT_PATCH_SIZE), name='CT')

# Model_3D=Unet3D(inputs,num_classes=3)

# Model_3D.compile(optimizer=OPTIMIZER, loss=[dice_loss], metrics=['accuracy',dice_coe])

# Model_3D.summary()
def my_generator(x_train, y_train, batch_size):

    data_generator = ImageDataGenerator(

            width_shift_range=0.1,

            height_shift_range=0.1,

            rotation_range=10,

            zoom_range=0.1).flow(x_train, x_train, batch_size)

    mask_generator = ImageDataGenerator(

            width_shift_range=0.1,

            height_shift_range=0.1,

            rotation_range=10,

            zoom_range=0.1).flow(y_train, y_train, batch_size)

    while True:

        x_batch, _ = data_generator.next()

        y_batch, _ = mask_generator.next()

        yield x_batch, y_batch
!pip install nilearn
from nilearn.plotting import view_img, glass_brain, plot_anat, plot_epi
# from tensorflow.keras.preprocessing.image import ImageDataGenerator



# image_batch, mask_batch = next(my_generator(X, y, 8))

# fix, ax = plt.subplots(8,2, figsize=(8,20))

# for i in range(8):

    

    

#     ax[i,0].imshow(image_batch[i,:,:,0])

#     ax[i,1].imshow(mask_batch[i,:,:,0])

# plt.show()

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint



weight_saver = ModelCheckpoint('lung.h5', monitor='val_dice_coef', 

                                              save_best_only=True, save_weights_only=True)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
# X.shape
# y.shape
# hist = Model_3D.fit(X, y,

#                            steps_per_epoch = 20,

                           

#                            epochs=10, verbose=2,

#                            )
# np.array([X[0]]).shape
# segmented=Model_3D.predict(np.array([X[0]]))
# segmented_=segmented[0,:,:,:,0]
# segmented_.shape
# import SimpleITK as sitk

# filtered_image = sitk.GetImageFromArray(segmented_)
# filtered_image 
# import nibabel as nib

# import numpy as np



# data = np.arange(4*4*3).reshape(4,4,3)



# new_image = nib.Nifti1Image(segmented_, affine=np.eye(4))



# new_image_ = nib.Nifti1Image(np.array(X[0]), affine=np.eye(4))





# plot_anat(new_image)

  

# view_img(new_image , new_image_)
# nib.save(new_image , '/kaggle/working/segmented.nii')
# nib.save(new_image_ , '/kaggle/working/original.nii')
!pip install medpy

import os

from os import listdir

from os.path import isfile, join

from medpy.io import load

import cv2

import numpy as np



def LoadLungData(x_shape, y_shape,limit):





    image_dir = '/kaggle/input/covid19-ct-scans/ct_scans/'

    label_dir = '/kaggle/input/covid19-ct-scans/infection_mask/'



    images = [f for f in listdir(image_dir) if (

        isfile(join(image_dir, f)) and f[0] != ".")]



    out = []

    count=0

    for f in images:

        

        count+=1

        if count<limit:

            #print('Count ',count)

            #print('Limit ',limit)



            image, _ = load(os.path.join(image_dir, f))

            label, _ = load(os.path.join(label_dir, f.replace('org_','')

                                         .replace('org_covid-19-pneumonia-','')

                                         .replace('covid-19-pneumonia-','')

                                        .replace('-dcm','')))



            if image.shape[0]!=label.shape[0]:

               print('File and label with different shapes ',f)





            #image=image/255



            try:



                image = reshape(image, new_shape=( x_shape, y_shape,576))

                #print('Image shape ',image.shape)

                label = reshape(label, new_shape=( x_shape, y_shape,576))

            except:

                print('Error in file ',f)

                raise



            out.append({"image": image, "seg": label, "filename": f})

        else:

            break



 

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
# out=LoadLungData(64, 64,200)

# print(out.shape)


# import matplotlib.pyplot as plt

# im_fdata=img.get_fdata()









# plt.figure(figsize=(10,8))



# # Iterate and plot random images

# for i in range(30):

#     plt.subplot(5, 6, i + 1)



#     plt.imshow(out[0]['seg'][i,:,:])

#     plt.axis('off')

    

# # Adjust subplot parameters to give specified padding

# plt.tight_layout()  
# X=[out[i]['image'] for i in range(len(out))]

# y=[out[i]['seg'] for i in range(len(out))]
# X=np.array(X)

# y=np.array(y)



# from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
# len(X_train)
# len(X_test)
# initial_epoch_of_training=0

# TRAIN_CLASSIFY_LEARNING_RATE =1e-4

# OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)



# INPUT_PATCH_SIZE=(576,64,64, 1)

# with tpu_strategy.scope():

#     inputs = tf.keras.Input(shape=(INPUT_PATCH_SIZE), name='CT')

#     Model_3D=Unet3D(inputs,num_classes=3)

#     Model_3D.compile(optimizer=OPTIMIZER, loss=[dice_loss], metrics=['accuracy',dice_coe])

#     Model_3D.summary()
# history = Model_3D.fit(X_train, y_train,

#                            batch_size=2,

#                            validation_data=(X_test,y_test),

#                            epochs=50, verbose=2,

#                            )

# auc=max(history.history['dice_coe'])







# df_history=pd.DataFrame.from_dict(history.history)

# df_history.to_csv('/kaggle/working/singleinput'+'.csv')



# from matplotlib.pyplot import figure

# figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')



# plt.plot(history.history['dice_coe'])

# plt.plot(history.history['val_dice_coe'])

# plt.title('Dice score single input')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['dice', 'val_dice'], loc='upper left')

# plt.savefig('/kaggle/working/singleinput.png')

# plt.show()

def cluster(img):

            vectorized = img.reshape((-1,4))

            vectorized = np.float32(vectorized)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            K = 4

            attempts=10

            ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

            center = np.uint8(center)

            res = center[label.flatten()]

            result_image = res.reshape((img.shape))

            return result_image
from scipy import ndimage

from skimage import filters
from skimage.morphology import disk



def reshape_cluster2(image, new_shape):

  

    reshaped_image = np.zeros(new_shape)



    print(reshaped_image.shape)

    print(image.shape)

    range_0=reshaped_image.shape[0]-image.shape[0]

    range_1=reshaped_image.shape[1]-image.shape[1]

    range_2=reshaped_image.shape[2]-image.shape[2]

    

    # if ((range_0>=0) and (range_1>=0)) and  (range_2>=0):

    #reshaped_image[0:image.shape[0],0:image.shape[1],0:image.shape[2]]+=image

    for i in range(image.shape[2]):

        

        reshaped_image[:,:,i]=filters.median(cv2.resize(image[:,:,i], ( 64, 64)),disk(1))



    for i in range(reshaped_image.shape[1]):

        

        reshaped_image[:,i,:]=filters.median(reshaped_image[:,i,:],disk(1))

        

        

    for i in range(reshaped_image.shape[0]):

        

        reshaped_image[i,:,:]=filters.median(reshaped_image[i,:,:],disk(1))

    reshaped_image=reshaped_image.transpose(2,1,0)

    print('Image reshaped ',reshaped_image.shape)

    # else:

    #   raise Exception("Invalid file shape")

       

        

    return reshaped_image
from skimage.morphology import disk



def reshape_cluster(image, new_shape):

  

    reshaped_image = np.zeros(new_shape)



    print(reshaped_image.shape)

    print(image.shape)

    range_0=reshaped_image.shape[0]-image.shape[0]

    range_1=reshaped_image.shape[1]-image.shape[1]

    range_2=reshaped_image.shape[2]-image.shape[2]

    

    # if ((range_0>=0) and (range_1>=0)) and  (range_2>=0):

    #reshaped_image[0:image.shape[0],0:image.shape[1],0:image.shape[2]]+=image

    for i in range(image.shape[2]):

        

        reshaped_image[:,:,i]=cluster(cv2.resize(image[:,:,i], ( 64, 64)))



    for i in range(reshaped_image.shape[1]):

        

        reshaped_image[:,i,:]=cluster(reshaped_image[:,i,:])

        

        

    for i in range(reshaped_image.shape[0]):

        

        reshaped_image[i,:,:]=cluster(reshaped_image[i,:,:])

    reshaped_image=reshaped_image.transpose(2,1,0)

    print('Image reshaped ',reshaped_image.shape)

    # else:

    #   raise Exception("Invalid file shape")

       

        

    return reshaped_image
from skimage.morphology import disk



def reshape_cluster_spectral(image, new_shape):

    

    

    import skimage.segmentation as seg







    reshaped_image = np.zeros(new_shape)



    print(reshaped_image.shape)

    print(image.shape)

    range_0=reshaped_image.shape[0]-image.shape[0]

    range_1=reshaped_image.shape[1]-image.shape[1]

    range_2=reshaped_image.shape[2]-image.shape[2]

    

    # if ((range_0>=0) and (range_1>=0)) and  (range_2>=0):

    #reshaped_image[0:image.shape[0],0:image.shape[1],0:image.shape[2]]+=image

    for i in range(image.shape[2]):

        0

        reshaped_image[:,:,i]= seg.slic(cv2.resize(image[:,:,i], ( 64, 64)),n_segments=30)



    for i in range(reshaped_image.shape[1]):

        

        reshaped_image[:,i,:]= seg.slic(reshaped_image[:,i,:],n_segments=30)

        

        

    for i in range(reshaped_image.shape[0]):

        

        reshaped_image[i,:,:]= seg.slic(reshaped_image[i,:,:],n_segments=30)

    reshaped_image=reshaped_image.transpose(2,1,0)

    print('Image reshaped ',reshaped_image.shape)

    # else:

    #   raise Exception("Invalid file shape")

       

        

    return reshaped_image
!pip install medpy

import os

from os import listdir

from os.path import isfile, join

from medpy.io import load

import cv2

import numpy as np



def LoadLungMaskData(x_shape, y_shape,limit):

    







    image_dir = '/kaggle/input/covid19-ct-scans/lung_and_infection_mask'

    label_dir = '/kaggle/input/covid19-ct-scans/infection_mask/'



    images = [f for f in listdir(image_dir) if (

        isfile(join(image_dir, f)) and f[0] != ".")]



    out = []

    count=0

    for f in images:

        

        count+=1

        if count<limit:

            #print('Count ',count)

            #print('Limit ',limit)



            image, _ = load(os.path.join(image_dir, f))

            label, _ = load(os.path.join(label_dir, f.replace('org_','')

                                         .replace('org_covid-19-pneumonia-','')

                                         .replace('covid-19-pneumonia-','')

                                        .replace('-dcm','')))



            if image.shape[0]!=label.shape[0]:

               print('File and label with different shapes ',f)





            #image=image/255



            try:



                image = reshape(image, new_shape=( x_shape, y_shape,576))

                #print('Image shape ',image.shape)

                label = reshape(label, new_shape=( x_shape, y_shape,576))

            except:

                print('Error in file ',f)

                raise



            out.append({"image": image, "seg": label, "filename": f})

        else:

            break



 

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
!pip install medpy

import os

from os import listdir

from os.path import isfile, join

from medpy.io import load

import cv2

import numpy as np



def LoadLungDataClusterData(x_shape, y_shape,limit):





    image_dir = '/kaggle/input/covid19-ct-scans/lung_and_infection_mask'

    label_dir = '/kaggle/input/covid19-ct-scans/infection_mask/'



    images = [f for f in listdir(image_dir) if (

        isfile(join(image_dir, f)) and f[0] != ".")]



    out = []

    count=0

    for f in images:

        

        count+=1

        if count<limit:

            #print('Count ',count)

            #print('Limit ',limit)



            image, _ = load(os.path.join(image_dir, f))

            label, _ = load(os.path.join(label_dir, f.replace('org_','')

                                         .replace('org_covid-19-pneumonia-','')

                                         .replace('covid-19-pneumonia-','')

                                        .replace('-dcm','')))



            if image.shape[0]!=label.shape[0]:

               print('File and label with different shapes ',f)





            #image=image/255



            try:



                image = reshape_cluster(image, new_shape=( x_shape, y_shape,576))

                #print('Image shape ',image.shape)

                label = reshape_cluster(label, new_shape=( x_shape, y_shape,576)).astype(int)

            except:

                print('Error in file ',f)

                raise



            out.append({"image": image, "seg": label, "filename": f})

        else:

            break



 

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
out=LoadLungData(64, 64,100)
out_=LoadLungDataClusterData(64, 64,100)

out__=LoadLungMaskData(64, 64,100)


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(30):

    plt.subplot(5, 6, i + 1)



    plt.imshow(out[0]['seg'][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(40,70,1):

    plt.subplot(5, 6, i + 1-40)



    plt.imshow(out_[0]['image'][i,:,:]+out[0]['image'][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(30):

    plt.subplot(5, 6, i + 1)



    plt.imshow(out__[0]['image'][i,:,:]+out[0]['image'][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(30):

    plt.subplot(5, 6, i + 1)



    plt.imshow(out[1]['image'][i,:,:]-out_[1]['image'][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


X=np.array([out[i]['image'] for i in range(len(out))])

y=np.array([out[i]['seg'] for i in range(len(out))])



import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(30):

    plt.subplot(5, 6, i + 1)



    plt.imshow(y[0][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  


import matplotlib.pyplot as plt

im_fdata=img.get_fdata()









plt.figure(figsize=(10,8))



# Iterate and plot random images

for i in range(30):

    plt.subplot(5, 6, i + 1)



    plt.imshow(X[0][i,:,:])

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()  
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print(X_train.shape)

print(y_train.shape)
import tensorflow as tf

initial_epoch_of_training=0

TRAIN_CLASSIFY_LEARNING_RATE =1e-4

OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)



INPUT_PATCH_SIZE=(576,64,64, 1)

with tpu_strategy.scope():

    inputs = tf.keras.Input(shape=(INPUT_PATCH_SIZE), name='CT')

    

    Model_3D=Unet3D(inputs,num_classes=3)

    Model_3D.compile(optimizer=OPTIMIZER, loss=[dice_loss], metrics=['accuracy',dice_coe])

    Model_3D.summary()


# history = Model_3D.fit(X_train, y_train,

#                            batch_size=2,

#                            validation_data=(np.array(X_test),np.array(y_test)),

#                            epochs=50, verbose=2,

#                            )

# df_history=pd.DataFrame.from_dict(history.history)

# df_history.to_csv('/kaggle/working/withoutsoby.csv')

# from matplotlib.pyplot import figure

# figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')



# plt.plot(history.history['dice_coe'])

# plt.plot(history.history['val_dice_coe'])

# plt.title('Dice score multi input')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['dice', 'val_dice'], loc='upper left')







Model_3D.load_weights('/kaggle/input/segmentationlung/single.h5')


for j in range(20):



    segmented=Model_3D.predict(np.array([X[j]]))

    segmented_=segmented[0,:,:,:,0]

    print(len(segmented_[segmented_>0]))



    import matplotlib.pyplot as plt

    im_fdata=img.get_fdata()









    plt.figure(figsize=(10,8))



    # Iterate and plot random images

    for i in range(0,30):

        plt.subplot(6, 6, i + 1)



        plt.imshow(segmented_[i,:,:])

        plt.axis('off')



    # Adjust subplot parameters to give specified padding

    plt.tight_layout()  

    plt.show()

    

    import nibabel as nib

    import numpy as np



    data = np.arange(4*4*3).reshape(4,4,3)



    new_image = nib.Nifti1Image(X[1], affine=np.eye(4))

    nib.save(new_image , '/kaggle/working/infection'+str(j)+'.nii')