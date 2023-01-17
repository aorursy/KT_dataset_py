import numpy as np # linear algebra

import pandas as pd # data processing, 

import cv2

# Input data files are available in the "../input/" directory

import os

import matplotlib.pyplot as plt

import itertools

# import segmentation_models as sm

import keras

import random

# from iteration_utilities import unique_everseen

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
#list(os.walk('/kaggle/input'))
os.listdir('/kaggle/')
os.listdir('/kaggle/input/')
os.listdir('/kaggle/input/severstal-steel-defect-detection/')
dir1 = '/kaggle/input/severstal-steel-defect-detection/'
df = pd.read_csv(dir1 + 'train.csv')

df.head()

#Each image is given along with its class id
df.tail(5)
len(os.listdir(dir1 + 'train_images/'))
len(os.listdir(dir1 + 'test_images/'))
def visualize_img():

    num = np.random.randint(0, len(os.listdir(dir1 + 'train_images/')))

    img = cv2.imread(dir1 + 'train_images/'+os.listdir(dir1 + 'train_images/')[num])

    print(img.shape)

    print(os.listdir(dir1 + 'train_images/')[num])

    plt.imshow(img)
visualize_img()
df.loc[df['ImageId_ClassId'].isin(['f380e604c.jpg_{}'.format(i) for i in range(1,5)])]
df[df['ImageId_ClassId'] == 'f380e604c.jpg_3']['EncodedPixels'].tolist()
df.head()
df['ImageId'] = [i.split('_')[0] for i in df['ImageId_ClassId'].tolist()]

df['ClassId'] = [i.split('_')[1] for i in df['ImageId_ClassId'].tolist()]
uniq_ids = list(np.unique(df['ImageId']))

len(uniq_ids)
df.head()
df['EncodedPixels'] = df['EncodedPixels'].replace(np.nan, 0)
df.head(8)
#Total images available for training are 12568 for each class

df['ClassId'].value_counts()
df1 = df[df['EncodedPixels']!=0]

df1.head(8)
df1['ClassId'].value_counts()
class_id_1 = df1[df1['ClassId'] == '1']['ImageId'].tolist()

class_id_2 = df1[df1['ClassId'] == '2']['ImageId'].tolist()

class_id_3 = df1[df1['ClassId'] == '3']['ImageId'].tolist()

class_id_4 = df1[df1['ClassId'] == '4']['ImageId'].tolist()
len(class_id_1),len(class_id_2),len(class_id_3),len(class_id_4)
all_defect_images = class_id_1 + class_id_2 + class_id_3 + class_id_4

defect_images = list(set(all_defect_images))

all_images = list(set(df['ImageId']))

non_defect_images = [i for i in all_images if i not in defect_images]
len(defect_images), len(non_defect_images)
train_images = defect_images[:5800] + non_defect_images[:100]

valid_images = defect_images[5800:6200] + non_defect_images[100:300]

print(len(train_images), len(valid_images))
random.seed(4)

random.shuffle(train_images)

random.shuffle(valid_images)
df1.info()
mask_imgs = df1['ImageId'].tolist()

print(len(mask_imgs))

print(len(np.unique(mask_imgs)))

# print(len(list(unique_everseen(mask_imgs))))

multi_masks = pd.Series(mask_imgs).value_counts() 

multi_masks = multi_masks[multi_masks > 1].index.tolist()

print(len(multi_masks))

# multi_masks
rle = df1[df1['ImageId'] == '0002cc93b.jpg']['EncodedPixels'].tolist()

rle
print(len(rle[0].split(' ')))

# rle[0].split(' ')
rle = list(map(int, rle[0].split(' ')))

# rle
pixel,pixel_count = [],[]

x23=[pixel.append(rle[i]) if i%2==0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]

# print('pixel starting points:\n',pixel)

# print('pixel counting:\n', pixel_count)
rle_pixels = [list(range(pixel[i],pixel[i]+pixel_count[i])) for i in range(0, len(pixel))]

# print('rle_pixels\n:', rle_pixels)
rle_mask_pixels = sum(rle_pixels,[]) 

# rle_mask_pixels = list(itertools.chain.from_iterable(rle_pixels))

# print('rle mask pixels:\n', rle_mask_pixels)
image = cv2.imread(dir1 + 'train_images/'+ '0002cc93b.jpg')

print('shape of image is:', image.shape)

plt.imshow(image)
def load_img_df(img):

    df2 = df[df['ImageId'] == img]

    return df2
def rle2mask(en_pix):

    en_pixels = [list(map(int, en_pix[0].split(' ')))]

    en_pixels = sum(en_pixels,[]) 

    pixel,pixel_count = [],[]

    [pixel.append(en_pixels[i]) if i%2==0 else pixel_count.append(en_pixels[i]) for i in range(0, len(en_pixels))]

    rle_pixels = [list(range(pixel[i],pixel[i]+pixel_count[i])) for i in range(0, len(pixel))]

    rle_mask_pixels = sum(rle_pixels,[]) 



    return rle_mask_pixels
def image_mask(img,image,j):

    l, b = image.shape[0], image.shape[1]

    df3 = load_img_df(img)

    en_pix = df3[df3['ClassId']==str(j)]['EncodedPixels'].tolist()

    mask = np.zeros((l,b), dtype=int)

#     print('en_pix',en_pix)

    if (en_pix == []) or (en_pix == [0]):

        return mask

    else:

        mask_img = np.zeros((l*b,1), dtype=int)

        rle_mask_pixels = rle2mask(en_pix)

        mask_img[rle_mask_pixels] = 1

        mask = np.reshape(mask_img, (b, l)).T

        return mask
def display_image_mask(img):

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize = (10,10)) 

    images = []

    image = cv2.imread(dir1 + 'train_images/' + img)

    for j in range(1,5):

        mask = image_mask(img,image,j)

        images.append(mask)

    images.append(1-sum(images))

    for im,ax in zip(images, ax.flatten()):

#         print(np.unique(im, return_counts=True))

        ax.imshow(im, cmap = 'gray')

#         print(np.unique(im, return_counts = True))
# for i in range(0, len(mask_imgs)//1000):

#     img55 = mask_imgs[i]

#     l3 = []

#     try:

#         display_image_mask(img = img55)

#     except:

#         l3.append(mask_imgs[i])
#Display image and masks for random image

img5 = mask_imgs[np.random.randint(0, len(mask_imgs))]

plt.title('input image {}'.format(img5))

plt.imshow(cv2.imread(dir1 + 'train_images/' + img5))  

display_image_mask(img = img5)
#Cross check mask class ids wrt dataframe

df[df['ImageId'] == img5]
def return_image_mask(img):

    image = cv2.imread(img)

    mask1 = np.zeros((256,1600,5), dtype = int)

    for j in range(1,5):

        mask = image_mask(img,image,j)

        mask1[:,:,j] = mask

    mask1[:,:,0] = 1-np.sum(mask1, axis=2)

    return image,mask1
import numpy as np 

import os

import skimage.io as io

import skimage.transform as trans

import numpy as np

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau

from keras.optimizers import SGD, RMSprop, Adadelta

from keras import backend as K

from keras import losses



checkpoint_path = '/kaggle/working/UNET_sep_28.hdf5'

weights_path = '/kaggle/working/UNET_sep_28.hdf5'

batch_size = 4

CLASSES = 5

sgd = SGD(lr=0.00146, decay=1e-6, momentum=0.9, nesterov=False)

checkpointer = ModelCheckpoint(monitor='loss',filepath=checkpoint_path,

                               verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001)
def unet(input_size = (256,1600,3)):

    inputs = Input(input_size)

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

    conv10 = Conv2D(5, 1, activation = 'softmax')(conv9)



    model = Model(input = inputs, output = conv10)

    return model
model = unet()
smooth = 1

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_multilabel(y_true, y_pred, numLabels=CLASSES):

    dice=0

    for index in range(numLabels):

        dice -= dice_coef(y_true[:,:,index], y_pred[:,:,index])

    return dice



def dice_coef_nd(y_true, y_pred):

    y_true_f = y_true.flatten()

    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return 2.*intersection, (np.sum(y_true_f)+np.sum(y_pred_f))



def dice_loss(y_true,y_pred):

    return K.constant(1.0) - dice_coef(y_true,y_pred)



def bce_dice_loss(y_true, y_pred):

    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss
model.compile(loss=bce_dice_loss,

                  optimizer=sgd,

                  metrics=[dice_coef])
def train_gen(images,batch_size):

    while True:

        indices = random.sample(range(0, len(images)), batch_size)

        all_images,all_masks = [],[]

        for i in indices:

#             print(images[i])

            im,op = return_image_mask(dir1+'train_images/'+images[i])

            all_images += [im]

            all_masks += [op]



        x = np.array(all_images)/255

        y = np.array(all_masks)

        yield (x,y)
model.fit_generator(train_gen(train_images, batch_size),

                    steps_per_epoch=len(train_images)//batch_size, epochs=20,

                   callbacks=[checkpointer,reduce_lr])

model.save_weights(checkpoint_path)
x = dir1 + 'test_images/' + os.listdir(dir1 + 'test_images/')[13]
model = unet()

model.load_weights(weights_path)

test = model.predict(np.expand_dims(cv2.imread(x), 0)/255)
test.shape

test1 = np.reshape(test, (256,1600,5))

np.unique(test1, return_counts = True)
agm = np.argmax(test1, axis = 2)

agm.shape

np.unique(agm, return_counts = True)
#Split into 5 channel mask after prediction

pred = np.zeros((256,1600,5), dtype = int)

for i in range(5):

    pred[:,:,i] = ((agm[:,:] == i)*255)
#Visualise results

fig1, ax1 = plt.subplots(nrows=5, ncols=1, figsize = (10,10)) 

for ij,axes in zip(range(5), ax1.flatten()):

#     print(axes)

    axes.imshow(pred[:,:,ij], cmap = 'gray')
fig2, ax2 = plt.subplots(nrows=5, ncols=1, figsize = (10,10)) 

for ij,axes in zip(range(5), ax2.flatten()):

    axes.imshow(pred[:,:,ij], cmap = 'gray')