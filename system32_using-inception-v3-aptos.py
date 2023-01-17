# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from sklearn.model_selection import train_test_split
# from tensorflow import set_random_seed
import tensorflow as tf
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input
from keras import backend as K
from sklearn.utils import shuffle

print(os.listdir('/kaggle/input'))
# print(os.listdir('/kaggle/input/inceptionv3/'))
SEED = 7
# np.random.seed(SEED)
# set_random_seed(SEED)
dir_path = "/kaggle/input/"
IMG_DIM = 299  # 224
BATCH_SIZE = 8
CHANNEL_SIZE = 3
NUM_EPOCHS = 60
TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'
FREEZE_LAYERS = 2  # freeze the first this many layers for training
CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
NUM_CLASSS = 5
ROOT_PATH = '/kaggle/input/aptos2019-blindness-detection'
TRAIN_PATH = '/kaggle/input/aptos2019-blindness-detection/' + TRAIN_DIR 
TEST_PATH = '/kaggle/input/aptos2019-blindness-detection/' + TEST_DIR 
dir_path = ROOT_PATH + '/'
def class_imbalance(df_train):    
    new_df = df_train[df_train['diagnosis']==0].sample(295,random_state = SEED)
    df1 = df_train[df_train['diagnosis']==1].sample(295,random_state = SEED)
    df2 = df_train[df_train['diagnosis']==2].sample(295,random_state = SEED)
    df4 = df_train[df_train['diagnosis']==4].sample(295,random_state = SEED)
    df3 = df_train[df_train['diagnosis']==3]
    new_df = new_df.append(df1,ignore_index = True)
    new_df = new_df.append(df2,ignore_index = True)
    new_df = new_df.append(df3,ignore_index = True)
    new_df = new_df.append(df4,ignore_index = True)
    return new_df
# print names of train images
train_img_names = glob.glob(TRAIN_PATH + '/*.png')
#print(train_img_names)

df_train = pd.read_csv(ROOT_PATH + '/train.csv')
df_train = class_imbalance(df_train)
df_train = shuffle(df_train)
df_train.head()
#print(df_train)


# print names of test images
test_img_names = glob.glob(TEST_PATH + '/*.png')
#print(test_img_names)
df_test = pd.read_csv(ROOT_PATH + '/test.csv')
#print(df_test)


def draw_img(imgs, target_dir, class_label='0'):
    for row in enumerate(imgs.iterrows()):
        name = row[1][1]['id_code'] + '.png'
        print(name)
        plt.figure(figsize=(15,10))
        img = plt.imread(dir_path + target_dir + '/' + name)
        plt.imshow(img)
        plt.title(class_label)
        plt.show()
        del img
        gc.collect
# Showing the class 0 image randomly
# CLASS_ID = 0
# draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])

# Showing the class 1 image randomly
# CLASS_ID = 1
# draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])


# Split Dataset

x_train, x_test, y_train, y_test = train_test_split(df_train.id_code, df_train.diagnosis, test_size=0.2,
                                                    random_state=SEED, stratify=df_train.diagnosis)




input_tensor = Input(shape = (299, 299, 3))

# create the base pre-trained model
base_model = InceptionV3(include_top=False, input_tensor=input_tensor, weights = 'imagenet')
# base_model.load_weights('/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# add a global spatial average pooling layer
x = base_model.output
output = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(NUM_CLASSS, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

for layer in model.layers:
    layer.trainable = True
    
print(model.summary())


epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print("available RAM:", psutil.virtual_memory())
gc.collect()
print("available RAM:", psutil.virtual_memory())

df_train.id_code = df_train.id_code.apply(lambda x: x + ".png")
df_test.id_code = df_test.id_code.apply(lambda x: x + ".png")
df_train['diagnosis'] = df_train['diagnosis'].astype('str')

import cv2
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
      
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
  #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
  #         print(img.shape)
        return img
def histogram_equalization(img_in):# segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])# calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
# mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')# merge the images in the three channels    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
    img_b = cdf_final_b[b]
  
    img_out = cv2.merge((img_b, img_g, img_r))# validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    return equ

def load_ben_color(image, sigmaX):
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = histogram_equalization(image)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image

def preprocess(image):
    return load_ben_color(image, sigmaX=30)
def clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def crop_image_from_gray(img,tol=59):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
        #         print(img.shape)
            return img

def something(img):
    # img = crop_image_from_gray(img)
    img = clahe_bgr(img)
    # image = histogram_equalization(image)
    # image = crop_image_from_gray(image)
    img = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX = 30) ,-4,0)
    return image
# Data Generator
train_datagen = image.ImageDataGenerator(rescale=1. / 255, 
                                         validation_split=0.15, 
                                         horizontal_flip=True,
                                         vertical_flip=True, 
                                         rotation_range=360, 
                                         zoom_range=0.2, 
                                         shear_range=0.1,
                                         preprocessing_function = preprocess)
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory= TRAIN_PATH + '/',
                                                    x_col='id_code',
                                                    y_col='diagnosis',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='training',
                                                    shaffle=True,
                                                    seed=SEED
                                                    )
valid_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory= TRAIN_PATH + '/',
                                                    x_col='id_code',
                                                    y_col='diagnosis',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='validation',
                                                    shaffle=True,
                                                    seed=SEED
                                                    )
#del x_train
# # del x_test
#del y_train
# del y_test
gc.collect()
#  color_mode= "grayscale",


NUB_TRAIN_STEPS = train_generator.n // train_generator.batch_size
NUB_VALID_STEPS = valid_generator.n // valid_generator.batch_size

NUB_TRAIN_STEPS, NUB_VALID_STEPS




eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving. 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',
                              verbose=1)


history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=NUB_TRAIN_STEPS,
                                     validation_data=valid_generator,
                                     validation_steps=NUB_VALID_STEPS,
                                     epochs=NUM_EPOCHS,
                                     #                            shuffle=True,  
                                     callbacks=[eraly_stop, reduce_lr],
                                     verbose=1)
gc.collect()
accu = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.plot(accu, label="Accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Acc', 'val_acc'])
plt.plot(np.argmax(history.history["val_accuracy"]), np.max(history.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.show()
history.history.keys()
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();