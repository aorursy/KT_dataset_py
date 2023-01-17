# imports

import os 

import numpy as np

import pandas as pd 

from mpl_toolkits.mplot3d import axes3d

from tqdm import tqdm_notebook 

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt

import cv2

import time

import seaborn as sns

import datetime



def timin():

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))

timin()
trainImgPath = "/kaggle/input/severstal-steel-defect-detection/train_images/"

trainCsv = "/kaggle/input/severstal-steel-defect-detection/train.csv"

dfFull = pd.read_csv(trainCsv)

dfFullEncodedOnly = dfFull[~dfFull['EncodedPixels'].isnull()]# get only image with labeled data for defects

print(dfFullEncodedOnly.shape)

print(dfFull.shape)

timin()
from skimage.io import imread

from scipy.ndimage.filters import convolve

emboss_kernel = np.array([  [0, 0, 0],

                            [0, 0, 1],

                            [0, 1, 1]])



emboss_kernel1 = np.array([ [1, 1, 0],

                            [1, 0, 0],

                            [0, 0, 0]])

                             

edge_kernel = np.array([    [-1,-1,-1],

                            [-1,4, -1],

                            [-1, -1, -1]])



horizontal  = np.array([    [-1,-2,-1],

                            [0, 0, 0],

                            [1, 2, 1]])

                             

vertSobel = np.array([      [-1,0,1],

                            [-2, 0, 2],

                            [-1, 0, 1]])



sharp = np.array([          [-1/9,-1/9,-1/9],

                            [-1/9, 1, -1/9],

                            [-1/9,-1/9, -1/9]])



edgeexce = np.array([        [1,1,1],

                            [1, -7, 1],

                            [1,1, 1]])



confilter = np.zeros((3,3,3))





mul = np.multiply(np.transpose(horizontal),np.transpose(vertSobel))



# have fun convolving on different channels

def convolves(image_copy):

    for i in range(0,3):

        image_copy[:,:,i] = convolve(image_copy[:,:,i], edgeexce)

        image_copy[:,:,i] = convolve(image_copy[:,:,i], edge_kernel)

    return image_copy

#x, y =next(getRandomBatch(4,validation_data=False))   

#plt.figure(figsize=(35,10))

#plt.imshow(x[0], cmap = 'Greys', interpolation = 'bicubic')

#plt.figure(figsize=(35,10))

#plt.imshow(y[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')



import cv2 

import numpy as np 

from skimage.io import imread

def drawContours(image):

    edged = cv2.Canny(image, 230, 240) 

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    cv2.drawContours(image, contours, -1, (0, 255, 0),1) 

    return image

plt.figure(figsize=(35,10))

plt.imshow(drawContours(cv2.imread(trainImgPath+"0002cc93b.jpg")), cmap = 'Greys', interpolation = 'bicubic')
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import random

import threading

trainImgPath = "/kaggle/input/severstal-steel-defect-detection/train_images/"

trainCsv = "/kaggle/input/severstal-steel-defect-detection/train.csv"

df1 = pd.read_csv(trainCsv)

df2 = df1[~df1['EncodedPixels'].isnull()].head(7000)

df3 = df1[df1['EncodedPixels'].isnull()].head(200)

df1 = pd.concat([df2,df3])

df1['ImageId'] = df1['ImageId_ClassId'].apply(lambda s:s.split("_")[0])

df1['Labels'] =  df1['ImageId_ClassId'].apply(lambda s:int(s.split("_")[1]))

df1.sample(frac=1)



getmask  = lambda x: getMaskByClass(x.EncodedPixels, x.Labels)

getimage = lambda img: cv2.resize(cv2.imread(trainImgPath+img),(800,128))



timin()

class ThreadSafeDataGenerator:

    def __init__(self, it):

        self.it = it

        self.lock = threading.Lock()



    def __iter__(self):

        return self



    def __next__(self):

        with self.lock:

            return self.it.__next__()



def safeItrWrap(f):

    def g(*a, **kw):

        return ThreadSafeDataGenerator(f(*a, **kw))

    return g



def getDataSlice(labelPassed,  batch_size1, validation_data):

    df = df1.copy()

    if labelPassed is not None:

        df = df[df['Labels']==labelPassed]

    if validation_data:

        randIndex = int(random.randint(df.shape[0]//1.7,df.shape[0] - 70))

        batch_size1=batch_size1*2

    else:

        randIndex = random.randint(0,df.shape[0]//1.5) 

    dfSlice = df.iloc[randIndex:randIndex+batch_size1].copy()

    dfSlice.drop(columns="ImageId_ClassId", inplace=True)

    return dfSlice



def getMaskByClass(listEncodedString, listLabels):

    mask = np.zeros((256, 1600, 4), dtype=np.int8)

    for encodedString,labels in zip (listEncodedString, listLabels):

        if len(str(encodedString))==0:

            mask[:,:,labels-1] =  np.zeros((256, 1600), dtype=np.int16)

        else:

            encodedString = str(encodedString).split(" ")

            flatmask = np.zeros(1600*256, dtype=np.int8)

            for i in range(0,len(encodedString)//2):

                start = int(encodedString[2*i])

                end = int(encodedString[2*i]) +int(encodedString[2*i+1])

                flatmask[start:end-1] =  1

            mask[:,:,labels-1] = np.transpose(flatmask.reshape(1600,256))

    return mask



@safeItrWrap

def getRandomBatch(labelPassed=None, batch_size1=24, validation_data=False):

    while True:

        dfSlice = getDataSlice(labelPassed,  batch_size1, validation_data)

        dfAgg = dfSlice.groupby(['ImageId']).agg({'Labels':list, 'EncodedPixels':list}).reset_index()

        dfAgg["EncodedPixels"] = dfAgg.apply(getmask, axis=1)

        dfAgg = dfAgg.head(batch_size1)

        labels = np.array(dfAgg["EncodedPixels"].tolist()).reshape(dfAgg.shape[0],256,1600,4)

        data =  dfAgg.ImageId.apply(getimage)

        data = np.array(data.tolist(), dtype=np.int16)

        if labelPassed is not None:

            yield data, labels[:,:,:,labelPassed-1].reshape(dfAgg.shape[0],256,1600,1)

        else:

            yield data, labels



@safeItrWrap

def getRandomTestBatch( batch_size1=24):

    testImgPath = "/kaggle/input/severstal-steel-defect-detection/test_images/"

    k = os.listdir(testImgPath)

    while True:

        index = random.randint(0,len(k))

        data =[]

        for iimgh in k[index:index+batch_size1]:

            p = cv2.resize(cv2.imread(testImgPath+iimgh),(800,128))

            data.append(p)

        yield np.array(data).reshape(batch_size1, 800,128,3)

          

            

            

x, y =next(getRandomBatch(1,validation_data=False))   

timin()

plt.figure(figsize=(35,10))

plt.imshow(x[0], cmap = 'Greys')

plt.figure(figsize=(35,10))

plt.imshow(y[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

#plt.figure(figsize=(35,10))

#plt.imshow(y[0,:,:,1], cmap = 'Greys', interpolation = 'bicubic')

#plt.figure(figsize=(35,10))

#plt.imshow(y[0,:,:,2], cmap = 'Greys', interpolation = 'bicubic')

#plt.figure(figsize=(35,10))

#plt.imshow(y[0,:,:,3], cmap = 'Greys', interpolation = 'bicubic')

timin()

aug  =  ImageDataGenerator(



                                             brightness_range=(0.8,1.2), 

                                            # shear_range=0.2, 

                                            # channel_shift_range=0.2, 

                                             fill_mode='nearest', 

                                            # cval=0.0, 

                                             horizontal_flip=True, 

                                            # vertical_flip=True, 

                                             rescale=1. / 255, 

                                         #    preprocessing_function=None, 

                                           #  data_format=None, 

                                           #  validation_split=0.0, 

                                             dtype=np.int16)
from keras.losses import binary_crossentropy

import tensorflow as tf

from keras import backend as K



def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def bce_dice_loss(y_true, y_pred):

    return 2 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)



def diff(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    return K.abs( K.sum( y_true_f-y_pred_f))





from keras.models import Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, concatenate, DepthwiseConv2D

from keras import optimizers

from keras.layers.normalization import BatchNormalization 

from keras.regularizers import l2

from keras.layers import Lambda, Reshape, Add, AveragePooling2D, MaxPooling2D, Concatenate, SeparableConv2D

from keras.initializers import RandomUniform

const1 = tf.convert_to_tensor (  np.full((16,128,800,24), 10) )



def antirectifier(x):

    x = np.where(x[0,:,:,0]>0.5,1,0)

    return x



def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, dilation_rate=(1,1)):

    """Function to add 2 convolutional layers with the parameters passed to it"""

    # first layer

    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size),\

              kernel_initializer = RandomUniform(minval=-1.1, maxval=1.1, seed=4), padding = 'same', dilation_rate=dilation_rate, kernel_regularizer=l2(0.0003))(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('tanh')(x)

    

    # second layer

    

    x =Conv2D(filters = n_filters, kernel_size = ( kernel_size, kernel_size),\

              kernel_initializer =  RandomUniform(minval=-1.2, maxval=1.2, seed=4), dilation_rate=dilation_rate, kernel_regularizer=l2(0.0003), padding = 'same')(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('tanh')(x)

    

    return x





def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, dilation_rate=(1,1)):

    # Contracting Path

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size =8, batchnorm = batchnorm, dilation_rate=dilation_rate)

    p1 = AveragePooling2D((2, 2))(c1)

    p1 = Dropout(dropout)(p1)

    

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 4, batchnorm = batchnorm)

    p2 = AveragePooling2D((2, 2))(c2)

    p2 = Dropout(dropout)(p2)

    

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 2, batchnorm = batchnorm)

    p3 = MaxPooling2D((2, 2))(c3)

    p3 = Dropout(dropout)(p3)

    

   

    c5 = conv2d_block(p3, n_filters = n_filters * 16, kernel_size = 2, batchnorm = batchnorm)

    

    # Expansive Path

   

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c5)

    u7 = concatenate([u7, c3])

    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 2, batchnorm = batchnorm)

    

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)

    u8 = concatenate([u8, c2])

    u8 = Dropout(dropout)(u8)

    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 4, batchnorm = batchnorm)

    

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)

    u9 = concatenate([u9, c1])

    u9 = Dropout(dropout)(u9)

    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 8, batchnorm = batchnorm)

    

    u10 = UpSampling2D()(c9)

   # cl1 = Lambda(antirectifier)(u10)





    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u10)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


from keras.layers import Dense, Activation, Conv2D, Input

from keras.layers.normalization import BatchNormalization as BN

from keras.optimizers import Adam

from keras import layers

import tensorflow as tf

import os

import datetime

from tensorflow.keras.metrics import TruePositives, TrueNegatives

import time

import gc

gc.collect()

from tensorflow import set_random_seed

set_random_seed(7)

epoches = 20

dilation_rate=(1,1)

learning_rate = 0.005

sep = 16 

vs = 24



from keras.models import load_model

import numpy as np

import pandas as pd

if (True):

    timin()

    model1 = load_model('../input/noaugs/model1111.h5',custom_objects={'diff': diff})

    timin()

    model2 = load_model('../input/noaugs/model2111.h5',custom_objects={'diff': diff})

    timin()

    model3 = load_model('../input/noaugs/model3111.h5',custom_objects={'diff': diff})

    timin()

    model4 = load_model('../input/noaugs/model4111.h5',custom_objects={'diff': diff})    

    timin()





if(False):

    model1 = get_unet(Input(shape=(128, 800, 3),dtype='float32'),dilation_rate=(2,2))

    model1.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy'])





    model2 = get_unet(Input(shape=(128, 800, 3)),dilation_rate=(2,2))

    model2.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy'])

    

    model3 = get_unet(Input(shape=(128, 800, 3)),  dilation_rate=dilation_rate)

    model3.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy'])



    model4 = get_unet(Input(shape=(128, 800, 3)), dilation_rate=(5,5))

    model4.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy'])





if(False):

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))



    history1 = model1.fit_generator(aug.flow(next(getRandomBatch(1,validation_data=False)), batch_size = 24),

                                    steps_per_epoch=sep, 

                                    epochs=epoches*1, 

                                    verbose=1, callbacks=None, 

                                    validation_data=getRandomBatch(1, validation_data=True),

                                    validation_steps=vs,  

                                    workers=4)

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))



    history2 = model2.fit_generator(aug.flow(next(getRandomBatch(2,validation_data=False)), batch_size = 16),

                                    steps_per_epoch=sep, 

                                    epochs=epoches*1, 

                                    verbose=0, callbacks=None, 

                                    validation_data=getRandomBatch(2, validation_data=True),

                                    validation_steps=vs,  

                                    workers=4)

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))



    history3 = model3.fit_generator(aug.flow(next(getRandomBatch(3,validation_data=False)), batch_size = 16),

                                    steps_per_epoch=sep, 

                                    epochs=epoches*4, 

                                    verbose=0, callbacks=None, 

                                    validation_data=getRandomBatch(3, validation_data=True),

                                    validation_steps=vs*2,  

                                    workers=4)

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))



    history4 = model4.fit_generator(aug.flow(next(getRandomBatch(4,validation_data=False)), batch_size = 16),

                                    steps_per_epoch=sep, 

                                    epochs=epoches*2, 

                                    verbose=0, callbacks=None, 

                                    validation_data=getRandomBatch(4, validation_data=True),

                                    validation_steps=vs,  

                                    workers=4)

    ts = time.time()

    st = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    print (st.replace(" ","-"))

if (False):

    model1.save("model1111.h5")

    model2.save("model2111.h5")

    model3.save("model3111.h5")

    model4.save("model4111.h5")
import random

import cv2

import numpy as np

import urllib.request

import os

submission =  pd.read_csv("/kaggle/input/severstal-steel-defect-detection/sample_submission.csv")

testImgPath = "/kaggle/input/severstal-steel-defect-detection/test_images/"

k = os.listdir(testImgPath)



i =0 



models = [model1,model2,model3,model4]

labels = ["1","2","3","4"]

labelsInt = [1,2,3,4]

results = []



def mask2rle(img):

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def getResults(x, image,models):

    inputImg = np.array(cv2.resize(cv2.imread(testImgPath+image),(800,128))).reshape(128, 800, 3).reshape(1, 128, 800, 3)

    prediction = models[x-1].predict(inputImg,batch_size=None, verbose=0, steps=None)

    return  np.where(cv2.blur(cv2.blur(prediction[0,:,:,0],(5,5)),(5,5))>0.9991,1,0)



def getResult(image):

    return [getResults( x, image, models) for x in labelsInt]

    



for image in k:

    results =  getResult(image)

    i = i +1

    for l,r in zip(labels, results):

        if (len(r))>0:

            submission.loc[submission['ImageId_ClassId']==image+"_"+l,["EncodedPixels"]] =  mask2rle(r).strip()

    if (i%50==0):

        print("done",i)



       



submission.to_csv("submission.csv",index=False)

img = cv2.imread(testImgPath+k[0]) 

print(np.array(img).shape)

x, y =next(getRandomBatch(3,validation_data=False))   

print(x[0].shape)

cv2.line(x[0],(0,64),(800,64),(255,255,255),5)

cv2.line(x[0],(400,0),(400,128),(255,255,255),5)

cv2.line(x[0],(0,84),(800,84),(0,0,0),5)

cv2.line(x[0],(600,0),(600,128),(0,0,0),5)

predict = model1.predict(x,batch_size=None, verbose=0, steps=None)

predict2 = model2.predict(x,batch_size=None, verbose=0, steps=None)

predict3 = model3.predict(x,batch_size=None, verbose=0, steps=None)

predict4 = model4.predict(x,batch_size=None, verbose=0, steps=None)

plt.figure(figsize=(35,10))

plt.imshow(x[0], cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(y[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(predict[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(predict4[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(predict2[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(predict3[0,:,:,0], cmap = 'Greys', interpolation = 'bicubic')

#clss = [1,2,3,4]

#modelss = [model1,model2,model3,model4]

#for m in modelss:

#    for c in clss:

#        print("Evaluate for model",c, m)

#        print(m.evaluate_generator(getRandomBatch(c,validation_data=False), steps=16,   verbose=1))
