import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from skimage.transform import resize
import tqdm
from tqdm import tqdm_notebook,tnrange
from keras.layers import Input,BatchNormalization,Dense,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D,Conv2D,UpSampling2D,Conv2DTranspose,MaxPool2D
from keras.models import Model,load_model,Sequential
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121,preprocess_input as densenet_preprocessing
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input as inception_preprocessing
from keras.applications.resnet50 import ResNet50,preprocess_input as resnet_preprocessing
from keras.applications.xception import Xception,preprocess_input as xception_preprocessing
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input as inceptionresnet_prepross
from keras.preprocessing.image import ImageDataGenerator,array_to_img,load_img,img_to_array
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras import backend as K
import tensorflow as tf
import os
import sys
import seaborn as sn
from PIL import Image
pd.set_option('display.max_colwidth',100)
%matplotlib inline
import gc
gc.collect()
Train_Image_folder='../input/train/images/'
Train_Mask_folder='../input/train/masks/'
Test_Image_folder='../input/test/images/'
Train_Image_name=os.listdir(path=Train_Image_folder)
Test_Image_name=os.listdir(path=Test_Image_folder)
Train_Image_path=[]
Train_Mask_path=[]
Train_id=[]
for i in Train_Image_name:
    path1=Train_Image_folder+i
    path2=Train_Mask_folder+i
    id1=i.split(sep='.')[0]
    Train_Image_path.append(path1)
    Train_Mask_path.append(path2)
    Train_id.append(id1)
  

Test_Image_path=[]
Test_id=[]
for i in Test_Image_name:
    path=Test_Image_folder+i
    id2=i.split(sep='.')[0]
    Test_Image_path.append(path)
    Test_id.append(id2)
    
df_Train_path=pd.DataFrame({'id':Train_id,'Train_Image_path':Train_Image_path,'Train_Mask_path':Train_Mask_path})
df_Test_path=pd.DataFrame({'id':Test_id,'Test_Image_path':Test_Image_path})

df_depths=pd.read_csv('../input/depths.csv')
df_sub=pd.read_csv('../input/sample_submission.csv')
df_Train_path=df_Train_path.merge(df_depths,on='id',how='left')
df_Test_path=df_Test_path.merge(df_depths,on='id',how='left')
df_Test_path=df_sub.merge(df_Test_path,on='id',how='left')
print(df_Train_path.shape,df_Test_path.shape)
df_Train_path.head()
df_Test_path.head()
def read_image(path,img_height,img_width,img_chan):
    pixel=np.zeros((len(path), img_height, img_width, img_chan),dtype=np.float32)
    for n, p in tqdm_notebook(enumerate(path), total=len(path)):
        img = load_img(p)
        x = img_to_array(img)[:,:,1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        x=x/255
        pixel[n]=x
    return pixel

img_height=128
img_width=128
img_chan=1
Train_Image_pixel=read_image(df_Train_path.Train_Image_path,img_height,img_width,img_chan)
Train_Mask_pixel=read_image(df_Train_path.Train_Mask_path,img_height,img_width,img_chan)

print('Train Image shape: ',Train_Image_pixel.shape)
print('Train Mask shape: ',Train_Mask_pixel.shape)

# Get and resize test images
def read_image2(path,img_height,img_width,img_chan):
    pixel=np.zeros((len(path), img_height, img_width, img_chan),dtype=np.float32)
    sizes_test = []
    for n, p in tqdm_notebook(enumerate(path), total=len(path)):
        img = load_img(p)
        x = img_to_array(img)[:,:,1]
        sizes_test.append([x.shape[0], x.shape[1]])
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        x=x/255
        pixel[n]=x
    return pixel,sizes_test

Test_Image_pixel,sizes_test=read_image2(df_Test_path.Test_Image_path,img_height,img_width,img_chan)
print('Test Image shape: ',Test_Image_pixel.shape)
array_to_img(Train_Image_pixel[0])
array_to_img(Train_Mask_pixel[0])
array_to_img(Test_Image_pixel[0])
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
# Another method
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
X=Train_Image_pixel
Y=Train_Mask_pixel
test=Test_Image_pixel
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.20,random_state=42)
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(test.shape)
# Build U-Net model
inputs = Input((img_height, img_width, img_chan))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[mean_iou])
#model.summary()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
results = model.fit(X, Y,validation_split=0.1, batch_size=8, epochs=30,callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})
#pred_train = model.predict(X_train, verbose=1)
#pred_val = model.predict(X_val, verbose=1)
preds_test = model.predict(test, verbose=1)
# Create list of upsampled test masks
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm_notebook(enumerate(df_Test_path.Test_Image_path))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
rle=sub[0].values
df_sub.rle_mask=rle
df_sub.head()
df_sub.to_csv('sub1.csv',index=False)
input_layer = Input((img_height, img_width, 1))
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model2 = Model(input_layer, output_layer)
model2.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coef])
#model2.summary()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-2.h5', verbose=1, save_best_only=True)
results = model2.fit(X, Y,validation_split=0.1, batch_size=8, epochs=30,callbacks=[earlystopper, checkpointer])

model2 = load_model('model-tgs-salt-2.h5', custom_objects={'dice_coef': dice_coef})
preds_test2 = model2.predict(test, verbose=1)
# Create list of upsampled test masks
preds_test_upsampled2 = []
for i in tnrange(len(preds_test2)):
    preds_test_upsampled2.append(resize(np.squeeze(preds_test2[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
pred_dict2 = {fn[:-4]:RLenc(np.round(preds_test_upsampled2[i])) for i,fn in tqdm_notebook(enumerate(df_Test_path.Test_Image_path))}
sub2 = pd.DataFrame.from_dict(pred_dict2,orient='index')
rle2=sub2[0].values
df_sub.rle_mask=rle2
df_sub.head()
df_sub.to_csv('sub2.csv',index=False)