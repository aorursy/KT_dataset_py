# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import nibabel as nib

from glob import glob

from sklearn.model_selection import train_test_split

from skimage.util import montage

from warnings import warn

def montage_nd(in_img):

    if len(in_img.shape)>3:

        return montage(np.stack([montage_nd(x_slice) for x_slice in in_img],0))

    elif len(in_img.shape)==3:

        return montage(in_img)

    else:

        warn('Input less than 3d image, returning original', RuntimeWarning)

        return in_img

BASE_IMG_PATH='/kaggle/input/verse3/spine.nii/spine.nii/'
os.listdir('/kaggle/input/verse3/spine.nii/spine.nii/')
# show some of the files

all_images=glob(os.path.join(BASE_IMG_PATH,'IMG_*'))

print(len(all_images),' matching files found:',all_images[0])

train_paths, test_paths = train_test_split(all_images,  test_size = 0.25)

print(len(train_paths), 'training size')

print(len(test_paths), 'testing size')
test_paths
img_show=nib.load(all_images[3]).get_data()

img_show.shape

# (152, 152, 195)(210, 210, 292)(177, 177, 300)(512, 512, 38)
all_images[3]
mask_path='/kaggle/input/verse3/spine.nii/spine.nii/IMG_verse134.nii.gz'

mask_show=nib.load(mask_path).get_data()

mask_show.shape
plt.imshow(mask_show[:,:,30],cmap='gray')
def gen_chunk1(in_img, in_mask,batch_size = 5):

        #in_img=(in_img-in_img.mean())/in_img.std()

        in_mask=in_mask/in_mask.max()

        img_batch = []

        mask_batch = []

        img=[]

        mask=[]

        for _ in range(batch_size):

            s_idx = np.random.choice(range(in_img.shape[0]-96))

            s_idy = np.random.choice(range(in_img.shape[1]-96))

            s_idz = np.random.choice(range(in_img.shape[2]-32))

            img_batch += [in_img[s_idx:(s_idx+96),s_idy:(s_idy+96),s_idz:(s_idz+32)]]

            mask_batch +=[in_mask[s_idx:(s_idx+96),s_idy:(s_idy+96),s_idz:(s_idz+32)]]

            img=np.stack(img_batch, 0)

            mask=np.stack(mask_batch, 0)  #叠加新的一维

        return img,mask
#image,mask=read_all_slices(all_images)

all_images
mask_path_0 = '/kaggle/input/verse3/spine.nii/spine.nii/MASK_verse102.nii.gz'

img_0=nib.load(all_images[0]).get_data()

mask_0=nib.load(mask_path_0).get_data()

x0,y0=gen_chunk1(img_0,mask_0,batch_size =200)



mask_path_1= '/kaggle/input/verse3/spine.nii/spine.nii/MASK_verse063.nii.gz'

img_1=nib.load(all_images[1]).get_data()

mask_1=nib.load(mask_path_1).get_data()

x1,y1=gen_chunk1(img_1,mask_1,batch_size =200)



y1.max(),x1.shape,y1.min()
mask_path_2= '/kaggle/input/verse3/spine.nii/spine.nii/MASK_verse064.nii.gz'

img_2=nib.load(all_images[2]).get_data()

mask_2=nib.load(mask_path_2).get_data()

mask_2=mask_2/mask_2.max()

#x2,y2=gen_chunk1(img_2,mask_2,batch_size = 5)

img_batch=[]

mask_batch=[]

for _ in range(200):

    s_idx = np.random.choice(range(img_2.shape[0]-96))

    s_idy = np.random.choice(range(img_2.shape[1]-96))

    s_idz = np.random.choice(range(img_2.shape[2]-32))

    img_batch += [img_2[s_idx:(s_idx+96),s_idy:(s_idy+96),s_idz:(s_idz+32)]]

    mask_batch +=[mask_2[s_idx:(s_idx+96),s_idy:(s_idy+96),s_idz:(s_idz+32)]]

    img=np.stack(img_batch, 0)

    mask=np.stack(mask_batch, 0)  #叠加新的一维

x2=img

y2=mask
y2.max(),x2.shape,y2.min()


mask_path_3= '/kaggle/input/verse3/spine.nii/spine.nii/MASK_verse134.nii.gz'

img_3=nib.load(all_images[3]).get_data()

mask_3=nib.load(mask_path_3).get_data()

x3,y3=gen_chunk1(img_3,mask_3,batch_size =200)
x3.shape,y3.shape
x=np.concatenate((x0,x1,x3),axis=0)

y=np.concatenate((y0,y1,y3),axis=0)

x.shape,y.shape
x1=x

y1=y

y1=np.expand_dims(y1,4)

x1=np.expand_dims(x1,4)

x1.shape,y1.shape
x_train, x_val,y_train,y_val = train_test_split(x1,y1,test_size = 0.25)

x_train.shape,y_train.shape
y_train.max()
z=x_val[20:21,:,:,:,0]

z=z[0]

print(z.shape)



y=y_val[20:21,:,:,:,0]

y=y[0]

print(y.shape)


plt.imshow(z[57,:,:])
plt.imshow(y[60,:,:],cmap='gray')
from skimage import transform
from keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D, Activation, Input

from keras.layers import MaxPooling3D, UpSampling3D, Deconvolution3D, concatenate,Add

from keras.models import Model

in_layer = Input((96,96, 32, 1))

bn = BatchNormalization()(in_layer)

cn1 = Conv3D(16, 

             kernel_size = (3, 3, 3),    #[32,128,128,8]

             padding = 'same',

             activation = 'linear')(bn)

bn1= Activation('relu')(BatchNormalization()(cn1))



x1=Conv3D(16, 

             kernel_size = (1, 1, 1),    #[32,128,128,8]

             padding = 'same',

             activation = 'linear')(bn)



cn2 = Conv3D(16, 

             kernel_size = (3, 3, 3),     #[32,128,128,8]

             padding = 'same',

             activation = 'linear')(bn1)

bn2 = Activation('relu')(Add()([BatchNormalization()(cn2),x1]))#[32,128,128,8]





dn1 = MaxPooling3D((2, 2, 2))(bn2)           #[16,64,64,8]





cn3 = Conv3D(32, 

             kernel_size = (3, 3, 3),

             padding = 'same',

             activation = 'linear')(dn1)     #[None,32,32,16]

bn3= Activation('relu')(BatchNormalization()(cn3))#[16,64,64,16]

x2=Conv3D(32, 

             kernel_size = (1, 1, 1),    

             padding = 'same',

             activation = 'linear')(dn1)        #[16,64,64,16]

cn4 = Conv3D(32, 

             kernel_size = (3, 3, 3),

             padding = 'same',

             activation = 'linear')(bn3)     #[16,64,64,16]

bn4 = Activation('relu')(Add()([BatchNormalization()(cn4),x2]))#[16,64,64,16]







dn2 = MaxPooling3D((2, 2, 2))(bn4)      #[8,32,32,16]





cn5 = Conv3D(64, 

             kernel_size = (3, 3, 3),

             padding = 'same',

             activation = 'linear')(dn2)    #[8,32,32,32]

bn5= Activation('relu')(BatchNormalization()(cn5))  #[8,32,32,32]



x3=Conv3D(64, 

             kernel_size = (1, 1, 1),  

             padding = 'same',

             activation = 'linear')(dn2)  #[8,32,32,32]



cn6 = Conv3D(64, 

             kernel_size = (3, 3, 3),

             padding = 'same',

             activation = 'linear')(bn5) 

bn6 = Activation('relu')(Add()([BatchNormalization()(cn6),x3]))  #[8,32,32,32]









up1 = Deconvolution3D(32, 

                      kernel_size = (3, 3, 3),

                      strides = (2, 2, 2),

                     padding = 'same')(bn6) #[16,64,64,16]



cat1 = concatenate([up1, bn4])            #[16,64,64,32]





ucn1 = Conv3D(32, 

             kernel_size = (3, 3, 3),    #[16,64,64,16]

             padding = 'same',

             activation = 'linear')(cat1)

ubn1= Activation('relu')(BatchNormalization()(ucn1))



ux1=Conv3D(32, 

             kernel_size = (1, 1, 1),  

             padding = 'same',

             activation = 'linear')(cat1)  #[16,64,64,16]

ucn2 = Conv3D(32, 

             kernel_size = (3, 3, 3),     

             padding = 'same',

             activation = 'linear')(ubn1) #[16,64,64,16]

ubn2 = Activation('relu')(Add()([BatchNormalization()(ucn2),ux1]))#[16,64,64,16]







up2 = Deconvolution3D(16, 

                      kernel_size = (3, 3, 3),

                      strides = (2, 2, 2),

                     padding = 'same')(ubn2)  #[32,128,128,8]



cat2 = concatenate([up2, bn2])             #[32,128,128,16]



ucn3 = Conv3D(16, 

             kernel_size = (3, 3, 3),    #[32,128,128,8]

             padding = 'same',

             activation = 'relu')(cat2)

ubn3= Activation('relu')(BatchNormalization()(ucn3))



ux2=Conv3D(16, 

             kernel_size = (1, 1, 1),  

             padding = 'same',

             activation = 'linear')(cat2)  #[32,128,128,8]



ucn4 = Conv3D(16, 

             kernel_size = (3, 3, 3),     

             padding = 'same',

             activation = 'linear')(ubn3)  #[32,128,128,8]

ubn4 = Activation('relu')(Add()([BatchNormalization()(ucn4),ux2]))



pre_out1 = Conv3D(8, 

             kernel_size = (3, 3, 3), 

             padding = 'same',

             activation = 'sigmoid')(ubn4)



pre_out1 = Conv3D(1, 

             kernel_size = (1, 1, 1), 

             padding = 'same',

             activation = 'sigmoid')(ubn4)



#pre_out1 = Cropping3D((2, 2, 2))(pre_out1) # avoid skewing boundaries

#out = ZeroPadding3D((2, 2, 2))(pre_out1)

sim_model = Model(inputs = [in_layer], outputs = [pre_out1])

sim_model.summary()
model=sim_model
model.summary()
from keras.optimizers import Adam

import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy



def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])

    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)
from keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics = [dice_coef,'mse'])

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('convlstm_model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



reduceLROnPlat =ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.000001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=20) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early]
model.fit(x_train,y_train,batch_size=10,

                    epochs=200,

                    validation_data=(x_val,y_val),

                    verbose=1,

                       callbacks = callbacks_list)
# binary_crossentropy 0.05656

# dice  0.21466

# val_loss did not improve from 0.20478
model.load_weights(weight_path)
model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics = [dice_coef,'mse'])

model.load_weights(weight_path)

model.fit(x_train,y_train,batch_size=10,

                    epochs=200,

                    validation_data=(x_val,y_val),

                    verbose=1,

                       callbacks = callbacks_list)

x_val.shape,y_val.shape

#x_val[10:11].shape,y_val[10:11].shape
xx=x_val[12:13]

yy=y_val[12:13]*24

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))

ax1.imshow(np.mean(xx[0, :, :, :, 0], 0),cmap='bone')

ax2.imshow(np.mean(xx[0, :, :, :, 0], 1),cmap='bone')

ax3.imshow(np.mean(xx[0, :, :, :, 0], 2),cmap='bone')
xx.shape,yy.shape
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))

ax1.imshow(np.mean(yy[0, :, :, :, 0], 0),cmap='gray')

ax2.imshow(np.mean(yy[0, :, :, :, 0], 1),cmap='gray')

ax3.imshow(np.mean(yy[0, :, :, :, 0], 2),cmap='gray')

yy.shape
#pred_seg = model.predict(np.expand_dims(test_single_vol,0))[0]
pred_seg = model.predict(xx)[0]

pred_seg=pred_seg*24

pred_seg.shape
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))

ax1.imshow(np.mean(pred_seg[:, :, :, 0], 0),cmap='gray')

ax2.imshow(np.mean(pred_seg[:, :, :, 0], 1),cmap='gray')

ax3.imshow(np.mean(pred_seg[:, :, :, 0], 2),cmap='gray')
x_val[0:1].shape
x_val=x_val[0,:,:,0]

x_val.shape
pred_seg.shape
pred_seg=pred_seg[:,:,:,0]

pred_seg.shape
plt.imshow(np.sum(pred_seg[::-1, :, :], 1), cmap = 'bone_r')
plt.imshow(np.sum(x_val[0:1][::-1, :, :, 0], 1), cmap = 'bone_r')
from skimage.util import montage

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))

ax1.imshow(np.max(xx[0,::-1, :, :, 0], 1), cmap = 'bone')

ax1.set_aspect(0.3)

ax2.imshow(np.sum(pred_seg[::-1, :, :, 0], 1), cmap = 'bone_r')

ax2.set_title('Prediction')

ax2.set_aspect(0.3)

ax3.imshow(np.sum(yy[0,::-1, :, :, 0], 1), cmap = 'bone_r')

ax3.set_title('Actual Lung Volume')

ax3.set_aspect(0.3)

fig.savefig('full_scan_prediction.png', dpi = 300)
pred_seg.shape
from skimage.util import montage

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))

ax1.imshow(np.max(xx[0,::-1, :, :, 0], 1), cmap = 'bone')

ax1.set_aspect(0.3)

ax2.imshow(np.sum(pred_seg[::-1, :, :, 0], 1), cmap = 'bone_r')

ax2.set_title('Prediction')

ax2.set_aspect(0.3)

ax3.imshow(np.sum(yy[0,::-1, :, :, 0], 1), cmap = 'bone_r')

ax3.set_title('Actual Lung Volume')

ax3.set_aspect(0.3)

fig.savefig('full_scan_prediction.png', dpi = 300)
pred_seg[::6, :, :, 0].shape
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))

ax1.imshow(montage(xx[0,::8, :, :, 0]), cmap = 'bone')

ax2.imshow(montage(pred_seg[::8, :, :, 0]), cmap = 'gray')

ax2.set_title('Prediction')

ax3.imshow(montage(yy[0,::8, :, :, 0]), cmap = 'gray')

ax3.set_title('Actual Mask')

fig.savefig('subsample_pred.png', dpi = 300)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))

bow_lung_idx = np.array([10,20]+[30, 40])

ax1.imshow(montage(xx[0,bow_lung_idx, :, :, 0]), cmap = 'bone')

ax2.imshow(montage(pred_seg[bow_lung_idx, :, :, 0]), cmap = 'bone_r')

ax2.set_title('Prediction')

ax3.imshow(montage(yy[0,bow_lung_idx, :, :, 0]), cmap = 'bone_r')

ax3.set_title('Actual Mask')

fig.savefig('bowel_vs_lung.png', dpi = 200)