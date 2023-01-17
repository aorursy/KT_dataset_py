import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input, GlobalMaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam

import keras
import tensorflow as tf

import random
def load_img(SL, train=True):
    """
    Given a list of image name, return the image array in the list
    """
    
    outlst=np.zeros((len(SL), h_size, w_size, channel))
    ori_size=np.zeros((len(SL), 2))
    for n, ID in enumerate(SL):
        if train==False:
            path=TEST_PATH+ID
        else:
            path=TRAIN_PATH+ID
        img=imread(path)
        ori_size[n, :]=img.shape[:2]   
        if img.ndim==3:
            img=resize(img, (h_size, w_size, 3))
        elif img.ndim==2:
            img=resize(img, (h_size, w_size))
            img=np.expand_dims(img, axis=-1)
            img[:,:,:]=img
            
        outlst[n]=img
    return outlst, ori_size

def check_id(image):
    ID=train['Id'][train['Image']==image].iloc[0]
    return ID

def flip_image(img):
    flipped=np.fliplr(img)
    return flipped
#define the input size. Since we are using VGG19 as training, we will set the size to be 224x224
h_size=224
w_size=224
channel=3

TRAIN_PATH="../input/whale-categorization-playground/train/"
TEST_PATH="../input/whale-categorization-playground/test/"

#Credit to "Michael Chmutov" for hand labling about 600 images that we can used to train our model
label=pd.read_csv('../input/bb-kaggle/bb_kaggle.csv')

#reformatting the imported table
label.rename(columns={'ystart xstart yend xend': 'coordinate'}, inplace=True)
ystart=[]
xstart=[]
yend=[]
xend=[]
for i in range (len(label)):
    a=label['coordinate'].iloc[i]
    a=a.split(' ')
    ystart.append(int(a[0]))
    xstart.append(int(a[1]))
    yend.append(int(a[2]))
    xend.append(int(a[3]))
label['ystart']=ystart
label['xstart']=xstart
label['yend']=yend
label['xend']=xend
label.head()
#define variables to store original size
ori_h=[]
ori_w=[]
ori_channel=[]
img_array=np.zeros((len(label), h_size, w_size, channel))
for n, images in enumerate(label['fn']):
    img=imread(TRAIN_PATH+images)
    ori_h.append(img.shape[0])
    ori_w.append(img.shape[1])
    _channel=img.ndim
    ori_channel.append(_channel)
    
    if _channel==3:
        img=resize(img, (h_size, w_size, channel))
    elif _channel==2:
        img=resize(img, (h_size, w_size))
        img=np.expand_dims(img, axis=-1)
        img[:,:,:]=img
    img_array[n]=img

label['original height']= ori_h
label['original width']= ori_w
label['original channel']= ori_channel

#Define and store y_ratio and x_ratio. these ratios are important so that we can transform the 224x224 bounding box back to its original size
y_ratio=label['original height']/h_size
x_ratio=label['original width']/w_size

t_ystart=np.rint(ystart/y_ratio)
t_xstart=np.rint(xstart/x_ratio)
t_yend=np.rint(yend/y_ratio)
t_xend=np.rint(xend/x_ratio)

label['t_ystart']=t_ystart
label['t_xstart']=t_xstart
label['t_yend']=t_yend
label['t_xend']=t_xend
y_true=np.concatenate((np.expand_dims(t_ystart.values, axis=-1), np.expand_dims(t_xstart.values, axis=-1), np.expand_dims(t_yend.values, axis=-1), np.expand_dims(t_xend.values, axis=-1)), axis=1)

label.head()

#credit to "taindow" for supplying additional labeled images. we will process the images as above as well.
label2=pd.read_csv('../input/bb-label-extra/bb_labels (1).csv')

y_ratio=label2['target_size_2']/h_size
x_ratio=label2['target_size_1']/w_size

t_ystart=np.rint(label2['ymin'].values/y_ratio)
t_xstart=np.rint(label2['xmin'].values/x_ratio)
t_yend=np.rint(label2['ymax'].values/y_ratio)
t_xend=np.rint(label2['xmax'].values/x_ratio)

label2['t_ystart']=t_ystart
label2['t_xstart']=t_xstart
label2['t_yend']=t_yend
label2['t_xend']=t_xend
y2_true=np.concatenate((np.expand_dims(t_ystart, axis=-1), np.expand_dims(t_xstart, axis=-1), np.expand_dims(t_yend, axis=-1), np.expand_dims(t_xend, axis=-1)), axis=1)

label2.head()
#load all images into array. All greyscale images will be converted to 3 channels as well
img2_array=np.zeros((len(label2), h_size, w_size, channel))
for n, images in enumerate(label2['image']):
    img=imread(TRAIN_PATH+images)
    _channel=img.ndim
       
    if _channel==3:
        img=resize(img, (h_size, w_size, channel))
    elif _channel==2:
        img=resize(img, (h_size, w_size))
        img=np.expand_dims(img, axis=-1)
        img[:,:,:]=img
    img2_array[n]=img
        
#sanity check
print (img_array.shape, img2_array.shape)
print (y_true.shape, y2_true.shape)
def check_distance(y_true, y_pred):
    """
    Customize loss function
    
    Argument:
    y_true-- Ground thruth position of the top left and bottom right corner of the bouding box.
    y_pred-- Predicted position of the top left and bottom right corner of the bounding box by the model.
    
    return-- the sum of distance between the top left and bottom right of y_true and y_pred
    """
    y11, x11, y12, x12=tf.split(y_true, 4, axis=1)
    y21, x21, y22, x22=tf.split(y_pred, 4, axis=1)
    
    dist_start=tf.sqrt(tf.square(y21-y11)+tf.square(x21-x11))
    dist_end=tf.sqrt(tf.square(y22-y12)+tf.square(x22-x12))
    sum_dist=dist_start+dist_end
    return sum_dist
    
def get_base_model():
    base_model=keras.applications.vgg19.VGG19(include_top=False, weights=None)
    base_model.load_weights('../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x=base_model.output
    x=GlobalMaxPooling2D()(x)
    x=Dropout(0.05)(x)
    dense_1=Dense(100)(x)
    dense_1=Dense(50)(dense_1)
    dense_1=Dense(10)(dense_1)
    base_output=Dense(4)(dense_1)
    base_model=Model(base_model.input, base_output)
    return base_model
    
in_dims=(h_size, w_size, channel)
inputs=Input(in_dims)
base_model=get_base_model()
outputs=base_model(inputs)

model=Model(inputs=inputs, outputs=outputs)
model.compile(loss=check_distance, optimizer=Adam(0.00001))
print (model.summary())



#concatenate all image array from both sources
all_img=np.concatenate([img_array, img2_array], axis=0)
all_y=np.concatenate([y_true, y2_true], axis=0)

"Uncomment to load pretrained weight"
#model.load_weights('../input/crop-image-weight/crop_image_v0.h5')
#model.evaluate(all_img, all_y)
"Uncomment to start training"
model.fit(all_img, all_y, batch_size=32, epochs=20, validation_split=0.2)
#model.save_weights('crop_image_v0.h5')
#Predicting bounding box of all train images
all_train_img_path=glob.glob('../input/whale-categorization-playground/train/*.jpg')
all_train_img_name=[] 
for file in all_train_img_path: 
    all_train_img_name.append(file.split('/')[-1])

_batch=2000 
num_batch=int(len(all_train_img_name)/_batch)
indexes=list(range(num_batch))
all_train_coordinateFrame=pd.DataFrame()
for i in indexes:
    print (f'Cropping image batch {i+1} of {indexes[-1]+1}')
    coordinateFrame=pd.DataFrame() 
    if i!=indexes[-1]:
        img_array, ori_size=load_img(all_train_img_name[i*_batch:(i+1)*_batch])
        coordinateFrame['image']=all_train_img_name[i*_batch:(i+1)*_batch]

    else: 
        img_array, ori_size=load_img(all_train_img_name[i*_batch:])
        coordinateFrame['image']=all_train_img_name[i*_batch:]

    coordinate=np.rint(model.predict(img_array))
    coordinateFrame['ori height']=ori_size[:, 0] 
    coordinateFrame['ori width']=ori_size[:, 1]
    
    y_ratio=coordinateFrame['ori height']/h_size
    x_ratio=coordinateFrame['ori width']/w_size
    
    coordinateFrame['ystart']=coordinate[:, 0]
    coordinateFrame['xstart']=coordinate[:, 1]
    coordinateFrame['yend']=coordinate[:,2]
    coordinateFrame['xend']=coordinate[:,3]
    
    #to make sure that no negatvie xstart and ystart
    coordinateFrame['xstart'][coordinateFrame['xstart']<0]=0 
    coordinateFrame['ystart'][coordinateFrame['ystart']<0]=0
    
    #transform the coordinate back to original shape, minus/add additional 10 pixels to avoid overcropping
    coordinateFrame['new ystart']=np.rint(coordinateFrame['ystart']*y_ratio)
    coordinateFrame['new xstart']=np.rint(coordinateFrame['xstart']*x_ratio)
    coordinateFrame['new yend']=np.rint(coordinateFrame['yend']*y_ratio)
    coordinateFrame['new xend']=np.rint(coordinateFrame['xend']*x_ratio)
    
    all_train_coordinateFrame=pd.concat([all_train_coordinateFrame, coordinateFrame], axis=0, ignore_index=True)
all_train_coordinateFrame.to_csv('whale_train_crop_table.csv')
all_train_coordinateFrame.head()
#Predicting bounding box of test images
all_test_img_path=glob.glob('../input/whale-categorization-playground/test/*.jpg')
all_test_img_name=[]
for file in all_test_img_path:
    all_test_img_name.append(file.split('/')[-1])

_batch=2000
num_batch=int(len(all_test_img_name)/_batch)
indexes=np.arange(num_batch+1)
all_test_coordinateFrame=pd.DataFrame()
for i in indexes:
    print (f'Cropping image batch {i+1} of {indexes[-1]+1}')
    coordinateFrame=pd.DataFrame() 
    if i!=indexes[-1]:
        img_array, ori_size=load_img(all_test_img_name[i*_batch:(i+1)*_batch], train=False)
        coordinateFrame['image']=all_test_img_name[i*_batch:(i+1)*_batch]
    else:
        img_array, ori_size=load_img(all_test_img_name[i*_batch:], train=False)
        coordinateFrame['image']=all_test_img_name[i*_batch:]

    coordinate=np.rint(model.predict(img_array))
    coordinateFrame['ori height']=ori_size[:, 0]
    coordinateFrame['ori width']=ori_size[:, 1]
    coordinateFrame['ystart']=coordinate[:, 0]
    coordinateFrame['xstart']=coordinate[:, 1]
    coordinateFrame['yend']=coordinate[:,2]
    coordinateFrame['xend']=coordinate[:,3] 
    coordinateFrame['xstart'][coordinateFrame['xstart']<0]=0
    coordinateFrame['ystart'][coordinateFrame['ystart']<0]=0

    y_ratio=coordinateFrame['ori height']/h_size
    x_ratio=coordinateFrame['ori width']/w_size

    coordinateFrame['new ystart']=np.rint(coordinateFrame['ystart']*y_ratio)
    coordinateFrame['new xstart']=np.rint(coordinateFrame['xstart']*x_ratio)
    coordinateFrame['new yend']=np.rint(coordinateFrame['yend']*y_ratio)
    coordinateFrame['new xend']=np.rint(coordinateFrame['xend']*x_ratio)
    all_test_coordinateFrame=pd.concat([all_test_coordinateFrame, coordinateFrame], axis=0, ignore_index=True)
all_test_coordinateFrame.head()
all_test_coordinateFrame.to_csv('whale_test_crop_table.csv')                                  

#sanity check: randomly choose 4 images from train to display before and after cropping
IMG_SHAPE=(224,224,3)
def crop_image(images, train=True):
    """take in images, crop according to train/test crop table, resize to IMG_SHAPE
    
    Arguments:
    images-- list consist of image name to be cropped
    train-- bool, True if the images are from train, False if test
    
    return:
    output_array-- array with axis-0 as number of images, axis-1 to 3 as cropped array of the input images
    """
    output_array=np.zeros((len(images), IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    for n, image in enumerate (images):
        if train:
            path=TRAIN_PATH+image
            table=all_train_coordinateFrame.copy()
        else:
            path=TEST_PATH+image
            table=all_test_coordinateFrame.copy()
            
        ystart=table['new ystart'][table['image']==image].values
        xstart=table['new xstart'][table['image']==image].values
        yend=table['new yend'][table['image']==image].values
        xend=table['new xend'][table['image']==image].values
        
        img_array=imread(path)
        height, width=img_array.shape[0], img_array.shape[1]
        
        #provide 10 pixel of margin, so that we are sure that entire flute will be cropped.
        ystart=int(max(ystart-10, 0))
        xstart=int(max(xstart-10, 0))
        yend=int(min(yend+10, height))
        xend=int(min(xend+10, width))
        
        #if image is greyscale, convert it to rgb with all channel = greyscale channel
        if img_array.ndim==3:
            img_array=img_array[ystart:yend, xstart:xend, :]
            img_array=resize(img_array, IMG_SHAPE)
        else:
            img_array=img_array[ystart:yend, xstart:xend]
            img_array=resize(img_array, (IMG_SHAPE[0], IMG_SHAPE[1]))
            img_array=np.expand_dims(img_array, axis=-1)
            img_array[:,:,:]=img_array
        output_array[n]=img_array
    return output_array

random_img=[random.choice(all_train_img_name) for _ in range(4)]

for n, img in enumerate(random_img):
    cropped_img=crop_image([img])
    
    plt.subplot(4,2,n*2+1)
    plt.imshow(load_img([img])[0][0])
    
    plt.subplot(4, 2, n*2+2)
    plt.imshow(cropped_img[0])

