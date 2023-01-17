#Necessary libraries are imported here
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
print(tf.__version__)
#Here we check is GPU is available for training or not Or whether the tensorflow version can utilize gpu 
physical_devices = tf.config.list_physical_devices('GPU') 
print("Number of GPUs :", len(physical_devices)) 
print("Tensorflow GPU :",tf.test.is_built_with_cuda())
if len(physical_devices)>0:
    device="/GPU:0"
else:
    device="/CPU:0"
fileList=[]
fl=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fileList.append(os.path.join(dirname, filename))
for i in fileList:
    if i.endswith('.jpg'):
        fl.append(i)
fileList=np.array(fl)
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
np.random.shuffle(fileList)
#The image size is kept at 400 x 400,
#and also we normalize the input image data so that each pixel value lies in 0~1 range
IMG_SIZE=(256,256)
batch=1
def decode_img(x):
    x=tf.io.read_file(x)
    x=tf.image.decode_jpeg(x,channels=3)
    x=tf.image.resize(x,IMG_SIZE)
    return x

def rgb_to_gs(x):
    gs = tf.image.rgb_to_grayscale(x)
    gs=tf.math.divide(gs,255)
    return gs


def create_dataset(filename_list):
    df=tf.data.Dataset.from_tensor_slices(filename_list)
    im=df.map(decode_img)
    gs=im.map(rgb_to_gs)
    df=tf.data.Dataset.zip((gs,im))
    df=df.batch(batch)
    df=df.prefetch(tf.data.experimental.AUTOTUNE)
    return df
#These are the actual objects representing the input pipeline for Training and Testing
train_df=create_dataset(fileList[:int(len(fileList)*0.8)])
test_df=create_dataset(fileList[int(len(fileList)*0.8):])
from tensorflow.keras.layers import Input,Conv2D,Add,Conv2DTranspose
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
model.compile(optimizer = tf.keras.optimizers.Adam(), loss ='MSE')
model.summary()
with tf.device(device):
    model.fit(train_df,
            validation_data=test_df,
              epochs=15)
model.save("ImageColourizationTFV2_256x256.h5")
ds_iter=iter(test_df)
for im_no in range(4):
    fig,axis=plt.subplots(1,3,figsize=(25, 80))
    x,y=next(ds_iter)
    yhat=model.predict(x)
    axis[0].imshow(x.numpy().reshape(256,256),cmap='gray')
    axis[0].axis('off')
    axis[0].set_title("Input Image", fontsize=30)
    axis[1].imshow(np.array(yhat[0],dtype=int).astype(int))
    axis[1].axis('off')
    axis[1].set_title("Model Prediction", fontsize=30)
    axis[2].imshow(np.array(y[0]).astype(int))
    axis[2].axis('off')
    axis[2].set_title("Original Image", fontsize=30)