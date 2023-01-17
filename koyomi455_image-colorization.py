import numpy as np 

import os

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Conv2D, Conv2DTranspose,UpSampling2D,Input ,Add

from tensorflow.keras.utils import plot_model

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU') 

print("Num GPUs:", len(physical_devices)) 

print(tf.test.is_built_with_cuda())

#tf.debugging.set_log_device_placement(True)
fileList=[]

fl=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fileList.append(os.path.join(dirname, filename))

for i in fileList:

    if i.endswith('.jpg'):

        fl.append(i)

fileList=np.array(fl)
verify=True

if verify:

    from PIL import Image

    fl=[]

    for filename in fileList:



        try:

          img = Image.open(filename)# open the image file

          img.verify() # verify that it is, in fact an image

          fl.append(filename)

        except Exception:

            print("Bad File",filename)

    fileList=np.array(fl)
IMG_SIZE=(256,256)

batch=8

def decode_img(x):

    x=tf.io.read_file(x)

    x=tf.image.decode_jpeg(x,channels=3)

    x=tf.image.resize(x,IMG_SIZE)

    return x

def rgb_to_gs(x):

    gs = tf.image.rgb_to_grayscale(x)

    gs=tf.math.divide(gs,255)

    return gs

def rgb_to_yuv(x):

    x=tf.image.rgb_to_yuv(x)

    return x



def create_dataset(filename_list):

    df=tf.data.Dataset.from_tensor_slices(filename_list)

    im=df.map(decode_img)

    gs=im.map(rgb_to_gs)

    im=im.map(rgb_to_yuv)

    df=tf.data.Dataset.zip((gs,im))

    df=df.shuffle(50)

    df=df.batch(batch)

    df=df.prefetch(tf.data.experimental.AUTOTUNE)

    return df
train_df=create_dataset(fileList[:int(len(fileList)*0.8)])

test_df=create_dataset(fileList[int(len(fileList)*0.8):])
ip=Input(shape=(256, 256,1))

x=Conv2D(128,5,padding='same',activation='relu')(ip)

x1=Conv2D(32,1,padding='same',activation='relu')(x)

x2=Conv2D(32,3,padding='same',activation='relu')(x)

x2=Conv2D(32,3,padding='same',activation='relu')(x2)





x=Add()([x2,x1])

x11=Conv2D(32,1,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x21)

x=Add()([x21,x11])

x11=Conv2D(32,1,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x21)

x=Add()([x21,x11])

x11=Conv2D(32,1,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x)

x21=Conv2D(32,3,padding='same',activation='relu')(x21)

x=Add()([x21,x11])



x=Conv2DTranspose(32,3,padding='same',activation='relu')(x)

x=Add()([x,x11])

x=Conv2DTranspose(32,3,padding='same',activation='relu')(x)



x=Add()([x,x1])

x=Conv2DTranspose(32,5,padding='same',activation='relu')(x)



x=Add()([x,x1])

x=Conv2DTranspose(64,7,padding='same',activation='relu')(x)

x=Conv2DTranspose(16,7,padding='same',activation='relu')(x)

x=Conv2DTranspose(64,7,padding='same',activation='relu')(x)

x=Conv2DTranspose(16,7,padding='same',activation='relu')(x)

x=Conv2DTranspose(3,7,padding='same')(x)

model=Model(inputs=ip,outputs=x)





model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.losses.MSE)

model.summary()
model.fit(train_df,

        validation_data=test_df,

          epochs=3)
model.save("ImageColourizationTFV2_256x256_XEpochs.h5")
im_no=8

x,y=next(iter(test_df))

yhat=model.predict(x)

fig,axis=plt.subplots(im_no,3,figsize=(30, 10*im_no))

for i in range(im_no):

    axis[i,0].imshow(np.array(x[i]).reshape(256,256),cmap='gray')

    axis[i,0].axis('off')

    rgb=cv2.cvtColor(np.float32(yhat[i]),cv2.COLOR_YUV2RGB)

    axis[i,1].imshow(np.array(rgb).astype(int))

    axis[i,1].axis('off')

    rgb=cv2.cvtColor(np.float32(y[i]),cv2.COLOR_YUV2RGB)

    axis[i,2].imshow(np.array(rgb).astype(int))

    axis[i,2].axis('off')