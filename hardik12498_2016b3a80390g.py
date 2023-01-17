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
import os
import cv2
import numpy as np
import tensorflow as tf
tf.test.is_gpu_available()
import numpy as np
import os
import shutil
files = np.genfromtxt("/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv",delimiter=",",dtype=np.str)[1:]
data_dir = "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/"

np.random.shuffle(files)
split = len(files) // 8

train_files = files[split:][:,0]
val_files = files[:split][:,0]


train_files = files[split:][:,0]
val_files = files[:split][:,0]


train_labels = files[split:][:,1].astype(np.int32)
val_labels = files[:split][:,1].astype(np.int32)

print(len(train_files),len(val_files))
print(len(train_labels),len(val_labels))

from matplotlib import pyplot as plt
print(np.unique(train_labels,return_counts=True))
print(np.unique(val_labels,return_counts=True))

def load_img(fname):
    img = cv2.imread(os.path.join(data_dir,fname))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(150,150))
    return img

imsize = (160,160,3)

np.unique(train_labels)
train_images = []
for i in train_files:
    img = load_img(i)
    train_images.append(img)
    
val_images = []
for i in val_files:
    img = load_img(i)
    val_images.append(img)
print(len(train_images),len(val_images))
std = np.std(train_images)
mean = np.mean(train_images)

print(std,mean)
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.SomeOf((2,5),
    [
        iaa.Noop(),
        iaa.Crop(percent=(0.05, 0.25)),
        iaa.Affine(
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180),
            shear=(-10, 10),
            order =  ia.ALL,
            mode='wrap',
        ),
            iaa.Noop(),
            iaa.OneOf([
                iaa.GaussianBlur((0, 2.0)),
                iaa.AverageBlur(k=(2, 5)),
                iaa.MedianBlur(k=(3, 5)),
            ]),
            iaa.Fliplr(),
            iaa.Sharpen(alpha=(0, 0.20), lightness=(0.75, 1.5)),
            iaa.Noop(),

            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.005*255), per_channel=0.1
            ),
            iaa.OneOf([
                iaa.Dropout((0.001, 0.01), per_channel=0.1),

            ]),
#             iaa.Invert(0.05, per_channel=True),
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Multiply((0.75, 1.25)),
            iaa.Noop(),
    ],
    random_order=True
)
from random import randint
from keras import layers
import keras

def augment_image(images):
    seq_det = seq.to_deterministic()
    images_aug = seq_det.augment_images(images)
    return images_aug
def generator(images,labels,aug=True,aug_res=True,batch_size = 64):
    images = np.array(images)
    labels = np.array(labels)
    while(True):
        p = np.random.permutation(len(images))
        images = images[p]
        labels = labels[p]
        for i in range(0,len(images)-batch_size,batch_size):
            try:
                X = []
                y = []
                for j in range(i,i+batch_size):
                    im = images[j]
                    
                    if(aug_res):
                        res = randint(80,256)
                        im = cv2.resize(im,(res,res))
                    X.append(im)
                    y.append(labels[j])
                y = keras.utils.to_categorical(y,num_classes=6)
                if(aug):
                    X = augment_image(X)
                for x in range(len(X)):
                    X[x] = cv2.resize(X[x],imsize[:2])
#                     X[x] = cv2.cvtColor(X[x],cv2.COLOR_RGB2HSV)
                    
                if(len(X) == 0):
                    continue
                X = (np.array(X,dtype=np.float32) - mean)/std
                yield X.reshape((-1,imsize[0],imsize[0],3)),y
            except:
                continue
batch_size = 10
train_gen = generator(train_images,train_labels,aug=True,aug_res=False,batch_size=batch_size)
val_gen = generator(val_images,val_labels,aug=False,aug_res=False,batch_size=batch_size)
x,y = next(train_gen)
print(x.shape)
# for i in range(len(x)):
#     plt.imshow(x[i])
#     plt.show()
#     print(y[i])
from keras import layers
import keras

keras.backend.clear_session()
classes = 6
def conv2d_block(x,filters,kernel_size,strides=1,padding='same',
                activation='relu',use_bias=True):
    x = layers.Convolution2D(filters,kernel_size,strides=strides,padding=padding,
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization()(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x


alpha = 1

img_input = layers.Input(imsize,name="input")
x = conv2d_block(img_input, int(64 * alpha), 5, padding='same')
x = layers.MaxPool2D(2,padding="same")(x)

alpha = alpha * 2
x = conv2d_block(x, int(64 * alpha), 5, padding='same')
x = conv2d_block(x, int(64 * alpha), 3, padding='same')
x = layers.MaxPool2D(2,padding="same")(x)
x = layers.BatchNormalization()(x)

alpha = alpha * 2
x = conv2d_block(x, int(64 * alpha), 5, padding='same')
x = conv2d_block(x, int(64 * alpha), 3, padding='same')
x = layers.MaxPool2D(2,padding="same")(x)
x = layers.BatchNormalization()(x)

x = conv2d_block(x, int(64 * alpha), 5, padding='same')
x = conv2d_block(x, int(64 * alpha), 3, padding='same')
x = layers.MaxPool2D(2,padding="same")(x)
x = layers.BatchNormalization()(x)

alpha = alpha * 2
x = conv2d_block(x, int(64 * alpha), 5, padding='same')
x = conv2d_block(x, int(64 * alpha), 3, padding='same')
x = layers.MaxPool2D(2,padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128,activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(classes, activation='softmax', name='predictions')(x)

model = keras.models.Model(img_input,x)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
for l in model.layers:
    try:
        l.kernel_regularizer = keras.regularizers.l2(0.0002) 
    except:
        print(l)
batch_size = 64

train_gen = generator(train_images,train_labels,aug=True,aug_res=True,batch_size=batch_size)
val_gen = generator(val_images,val_labels,aug=False,aug_res=False,batch_size=batch_size)

train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size
chkpt = keras.callbacks.ModelCheckpoint(filepath="M2model_st1.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=15
                    ,use_multiprocessing=True,workers=4,callbacks=[chkpt])

batch_size = 128
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=1e-3),metrics=["acc"])
train_gen = generator(train_images,train_labels,batch_size=batch_size)
val_gen = generator(val_images,val_labels,batch_size=batch_size,aug=False,aug_res=False)

train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size

chkpt = keras.callbacks.ModelCheckpoint(filepath="model_st3.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=10000
                    ,use_multiprocessing=True,workers=2,callbacks=[chkpt])
batch_size = 128
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=1e-4),metrics=["acc"])
train_gen = generator(train_images,train_labels,batch_size=batch_size)
val_gen = generator(val_images,val_labels,batch_size=batch_size,aug=False,aug_res=False)

train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size

chkpt = keras.callbacks.ModelCheckpoint(filepath="model_st3.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=10000
                    ,use_multiprocessing=True,workers=2,callbacks=[chkpt])
batch_size = 128
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=1e-4),metrics=["acc"])
train_gen = generator(train_images,train_labels,batch_size=batch_size,aug=False,aug_res=False)
val_gen = generator(val_images,val_labels,batch_size=batch_size,aug=False,aug_res=False)

train_steps = len(train_images) // batch_size
val_steps = len(val_images) // batch_size

chkpt = keras.callbacks.ModelCheckpoint(filepath="model_st4.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=10000
                    ,use_multiprocessing=True,workers=2,callbacks=[chkpt])
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.SomeOf((1,5),
    [
        iaa.Noop(),
        iaa.Crop(percent=(0.05, 0.15)),
        iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            shear=(-10, 10),
            order =  ia.ALL,
            mode='reflect',
        ),
            iaa.Noop(),
            iaa.Fliplr(),
            iaa.Flipud(),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.005*255), per_channel=0.1
            ),
#             iaa.Invert(0.05, per_channel=True),
            iaa.Multiply((0.75, 1.25)),
            iaa.Noop(),
    ],
    random_order=True
)

batch_size = 160
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.RMSprop(lr=1e-4),metrics=["acc"])
train_gen = generator(train_images,train_labels,batch_size=batch_size,aug_res=False)
val_gen = generator(val_images,val_labels,batch_size=batch_size,aug=False,aug_res=False)

train_steps = 1 + (len(train_images) // batch_size)
val_steps = 1 + (len(val_images) // batch_size)
# 
model.load_weights("/kaggle/working/model_st4.h5")
chkpt = keras.callbacks.ModelCheckpoint(filepath="model_st5.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=10000
                    ,callbacks=[chkpt])
batch_size = 160
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Nadam(lr=1e-4),metrics=["acc"])
train_gen = generator(train_images,train_labels,batch_size=batch_size,aug_res=False)
val_gen = generator(val_images,val_labels,batch_size=batch_size,aug=False,aug_res=False)

train_steps = 1 + (len(train_images) // batch_size)
val_steps = 1 + (len(val_images) // batch_size)
# 
model.load_weights("/kaggle/working/model_st5.h5")
chkpt = keras.callbacks.ModelCheckpoint(filepath="model_st6.h5",save_best_only=True)
model.fit_generator(train_gen,train_steps,validation_data=val_gen,validation_steps=val_steps,verbose=1,epochs=10000
                    ,callbacks=[chkpt])
path = "/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images/"

model.load_weights("model_st6.h5")
outf = open("submission.csv","w")
outf.write("image_name,label\n")
for f in os.listdir(path):
    img = cv2.imread(path+f)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,imsize[:2])
    img = (img - mean) / std
    lab = model.predict(img.reshape((1,160,160,3)))[0]
    lab = np.argmax(lab)
    outf.write(f + "," + str(lab) + "\n")
outf.close()
