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

# These are each binary classifiers for each mask

labels = ["1","2","3","4"]

labelsInt = [1,2,3,4]

trainImgPath = "/kaggle/input/severstal-steel-defect-detection/train_images/"

trainCsv = "/kaggle/input/severstal-steel-defect-detection/train.csv"

dfFull = pd.read_csv(trainCsv)

df1 = dfFull[~dfFull['EncodedPixels'].isnull()]# get only image with labeled data for defects

df1['ImageId'] = df1['ImageId_ClassId'].apply(lambda s:s.split("_")[0])

df1['Labels'] =  df1['ImageId_ClassId'].apply(lambda s:int(s.split("_")[1]))

df1.drop(columns="ImageId_ClassId", inplace=True)

import cv2

from skimage.io import imread

from scipy.ndimage.filters import convolve

import numpy as np 

# Conv kernels

embossLower = np.array([  [0, 0, 0],   [0, 0, 1],  [0, 1, 1]])

embossUpper = np.array([ [1, 1, 0],   [1, 0, 0],  [0, 0, 0]])

emboss = np.array([ [-2, -1, 0],   [-1, 1, 1],  [0, 1, 2]])

edge = np.array([    [-1,-1,-1],  [-1,4, -1], [-1, -1, -1]])

horizontal  = np.array([[-1,-2,-1],  [0, 0, 0],  [1, 2, 1]])

vertical = np.array([[-1,0,1],    [-2, 0, 2], [-1, 0, 1]])

sharp = np.array([[-1/9,-1/9,-1/9],   [-1/9, 1, -1/9],    [-1/9,-1/9, -1/9]])

edgex = np.array([[1,1,1],   [1, -7, 1],  [1,1, 1]])



class PreProcessor:



    kernels = None

    iterations = None



    

    def __init__(self,kernel, iteration):

        self.kernels= kernel

        self.iterations = iteration

        

    

    def conv(self,image):

        if (image.shape[2]==3):

            print("e")

            return self.convc(image,3)

        else:

            return self.convc(image,1)

            

    

    def convc(self, image, channels):

        for k,itr in zip(self.kernels, self.iterations):

            for i in range(0,itr):

                for c in range(0,channels):

                    image[:,:,c]  = convolve(image[:,:,c], k)

        return image
import cv2 

import numpy as np 

import os

import random

k = os.listdir(trainImgPath)

print( k[0])

from skimage.io import imread

def drawContours(image):

    PreProcessor([emboss],[1,1]).conv(image)*1.02

    edged = cv2.Canny(image, 120, 240) 

    return image, edged

plt.figure(figsize=(35,10))

p, y = drawContours(cv2.imread( trainImgPath+k[random.randint(0,90)]))

plt.imshow(p, cmap = 'Greys', interpolation = 'bicubic')

plt.figure(figsize=(35,10))

plt.imshow(y, cmap = 'Greys', interpolation = 'bicubic')
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import random

import threading

import os



def getMaskFromString(listEncodedString, listLabels):

    mask = np.zeros((256, 1600, 4), dtype=np.int8)

    for encodedString,labels in zip (listEncodedString, listLabels):

        if len(str(encodedString))==0:

            mask[:,:,labels-1] =  np.zeros((256, 1600), dtype=np.int16)

        else:

            encodedString = str(encodedString).split(" ")

            flatmask = np.zeros(1600*256, dtype=np.int8)

            for i in range(0,len(encodedString)//2):

                start = int(encodedString[2*i])

                end = start + int(encodedString[2*i+1])

                flatmask[start:end-1] =  1

            mask[:,:,labels-1] = np.transpose(flatmask.reshape(1600,256))

    return mask

#PreProcessor([emboss,sharp],[1,1]).conv(cv2.resize(cv2.imread(trainImgPath+img),(800,128)))*1.05



class SafePandasDG:

    

    """This is a utility data generator that utilizes pandas, opencv"""

    

    datapath =  "/kaggle/input/severstal-steel-defect-detection/train_images/"

    testpath =  "/kaggle/input/severstal-steel-defect-detection/test_images/"

    df = None

    dfo = None

    split = 1.7

    label = None

    index = 0

    batch = 0

    valid = False

    maskColumn =  "EncodedPixels"   

    dataColumn = 'ImageId'

    labelColumn = 'Labels'

    aggDict = {labelColumn:list, maskColumn:list}

    getMask =  lambda self, x: getMaskFromString(x[self.maskColumn], x[self.labelColumn])

    getImage = lambda self, x: cv2.resize(cv2.imread(self.datapath+x),(800,128))

    test = False

    pollute = False

    batchNum = 0

    alllabels = [1,2,3,4]



    def __init__(self, df, label=None, batch=24, valid=False, test=False):

        

        """Constructor agrumnents are dataframe, label for which data, if validation or test data """

        

        self.dfo = df

        self.label = label 

        self.batch = batch

        self.valid = valid

        self.test = test

        self.lock = threading.Lock()



    

    def __iter__(self):

        return self



    

    def __next__(self):

        with self.lock:

            if self.test == False:

                return self.getRandomBatch().__next__()

            else:

                return self.getRandomTestBatch().__next__()

    

    

    def getSliceMask(self, dfAgg):

        if self.pollute == True:

            return np.zeros((dfAgg.shape[0],256, 1600, 4), dtype=np.int8)

        dfAgg[self.maskColumn] = dfAgg.apply(self.getMask,axis=1)

        mask = dfAgg[self.maskColumn].tolist()

        mask = np.array(mask).reshape(dfAgg.shape[0],256,1600,4)

        return mask 

    

    

    

    def setRandomIndex(self):

        if self.valid:

            self.index = int( random.randint(self.df.shape[0] // self.split, self.df.shape[0] - self.batch))

        else:

            self.index = int( random.randint(0, self.df.shape[0] // self.split))

        

    

    def getDataByIndex(self):

        dfSlice = self.df.iloc[self.index: self.index + self.batch].copy()

        dfAgg = dfSlice.groupby([self.dataColumn]).agg(self.aggDict).reset_index()

        dfAgg = dfAgg.head(self.batch)

        data = dfAgg[self.dataColumn].apply(self.getImage)

        data = np.array(data.tolist(), dtype=np.int16)

        mask = self.getSliceMask(dfAgg)

        if self.label is  None:

            return data, mask

        else:

            return data, mask[:,:,:,self.label-1].reshape(dfAgg.shape[0],256,1600,1)

    

    

    def getRandomBatch(self):

        while True:

            self.batchNum = self.batchNum + 1

            if self.batchNum%3==0:

                self.pollute = True

            else:

                self.pollute = False

            if self.label is not None and self.pollute == False:

                self.df = self.dfo[ self.dfo[ self.labelColumn ] == self.label ]

            if self.label is not None and self.pollute == True:

                plabel  = [it for it in self.alllabels if it != self.label] 

                plabel  = plabel[random.randint(0,2)]

                self.df = self.dfo[ self.dfo[ self.labelColumn]  == plabel ]

            self.setRandomIndex()

            data, labels = self.getDataByIndex()

            yield  data, labels

            

    

    def getRandomTestBatch(self):

        testImages = os.listdir(self.testpath)

        self.index = int( random.randint(0,len(testImages)))

        while True:

            data = []

            for ind in range(self.index, self.index + self.batch):

                data.append(cv2.resize(cv2.imread(self.testpath+testImages[ind]),(800,128)))

            yield np.array(data).reshape(self.batch, 128, 800, 3), None

            

  

for label in range(1,5):

    x, y = next(SafePandasDG(df1, label, 1, False, False))   

    plt.figure(figsize=(35,10))

    plt.imshow(x[0], cmap = 'Greys')

    plt.figure(figsize=(35,10))

    plt.imshow(y[0,:,:,0]*255, cmap = 'Greys', interpolation = 'bicubic')



    

x, y = next(SafePandasDG(df1, None, 1, False, True))   

plt.figure(figsize=(35,10))

plt.imshow(x[0], cmap = 'Greys')

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

    return 2 * binary_crossentropy(y_true, y_pred)



def diff(y_true, y_pred):

    y_true_f = np.array(y_true).flatten()

    y_pred_f = np.array(y_pred).flatten()

    return np.abs(np.sum(  np.subtract(y_true_f,y_pred_f)))





from keras.models import Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, concatenate, DepthwiseConv2D

from keras import optimizers

from keras.layers.normalization import BatchNormalization 

from keras.regularizers import l2

from keras.layers import Lambda, Reshape, Add, AveragePooling2D, MaxPooling2D, Concatenate, SeparableConv2D, ELU

from keras.initializers import RandomUniform



def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, dilation_rate=(1,1)):

    """Function to add 2 convolutional layers with the parameters passed to it"""

    # first layer

    x = BatchNormalization()(input_tensor)

    x = Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size),\

              kernel_initializer ='he_normal', padding = 'same', dilation_rate=dilation_rate, kernel_regularizer=l2(0.00003))(x)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    

    # second layer

    x = BatchNormalization()(input_tensor)

    x = Conv2D(filters = n_filters, kernel_size = ( kernel_size, kernel_size),\

              kernel_initializer =  'he_normal', dilation_rate=dilation_rate, kernel_regularizer=l2(0.00003), padding = 'same')(x)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    

    return x





def get_unet(input_img, n_filters = 32, dropout = 0.1, batchnorm = True, dilation_rate=(1,1)):

    # Contracting Path

    c1 = conv2d_block(input_img, n_filters * 1, kernel_size =3, batchnorm = batchnorm, dilation_rate=dilation_rate)

    p1 = AveragePooling2D((2, 2))(c1)

    p1 = Dropout(dropout)(p1)

    

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 2, batchnorm = batchnorm)

    p2 = AveragePooling2D((2, 2))(c2)

    p2 = Dropout(dropout)(p2)

    

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 1, batchnorm = batchnorm)

    p3 = MaxPooling2D((2, 2))(c3)

    p3 = Dropout(dropout)(p3)

    



    c5 = conv2d_block(p3, n_filters = n_filters * 8, kernel_size = 1, batchnorm = batchnorm)



    # Expansive Path

   

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c5)

    u7 = concatenate([u7, c3])

    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 1, batchnorm = batchnorm)

    

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)

    u8 = concatenate([u8, c2])

    u8 = Dropout(dropout)(u8)

    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 2, batchnorm = batchnorm)

    

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)

    u9 = concatenate([u9, c1])

    u9 = Dropout(dropout)(u9)

    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    c10 = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    outputs = UpSampling2D()(c10)



    model = Model(inputs=[input_img], outputs=[outputs])

    return model
from keras.models import load_model

from keras.layers import Dense, Activation, Conv2D, Input

from keras.layers.normalization import BatchNormalization as BN

from keras.optimizers import Adam

from keras import layers

import tensorflow as tf

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

import os

import datetime

from tensorflow.keras.metrics import TruePositives, TrueNegatives

import time

import gc

gc.collect()

from tensorflow import set_random_seed

set_random_seed(2)

epoches = 50

dilation_rate=(1,1)

learning_rate = 0.0007

sep = 36

vs = 24



adams = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.000005, amsgrad=False)



if (True):

    timin()

    model1 = load_model('../input/oldmodel1/model1111.h5',custom_objects={'diff': diff,'dice_coef':dice_coef} )

    timin()

    model2 = load_model('../input/oldmodel1/model2111.h5',custom_objects={'diff': diff,'dice_coef':dice_coef} )

    timin()

    model3 = load_model('../input/oldmodel1/model3111.h5',custom_objects={'diff': diff,'dice_coef':dice_coef} )

    timin()

    model4 = load_model('../input/oldmodel1/model4111.h5',custom_objects={'diff': diff,'dice_coef':dice_coef} )    

    timin()

    





if(False):

    model1 = get_unet(Input(shape=(128, 800, 3),dtype='float32'), dilation_rate=(2,2))

    model1.compile(optimizer=adams, 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy',dice_coef])





    model2 = get_unet(Input(shape=(128, 800, 3)),dilation_rate=(2,2))

    model2.compile(optimizer=adams, 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy',dice_coef])

    

    model3 = get_unet(Input(shape=(128, 800, 3)),  dilation_rate=(4,4))

    model3.compile(optimizer=adams, 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy',dice_coef])



    model4 = get_unet(Input(shape=(128, 800, 3)), dilation_rate=(5,5))

    model4.compile(optimizer=adams, 

                   loss='binary_crossentropy', 

                   metrics=[diff,'binary_accuracy',dice_coef])



models = [model1,model2,model3,model4]

    

getBatch =   [ SafePandasDG(df1, i, 24, False) for i in range(1,5) ]

validBatch = [ SafePandasDG(df1, i, 24, True) for i in range(1,5) ] 



if(False):

    timin()

    history = [models[i-1].fit_generator(getBatch[i-1],

                                        steps_per_epoch=sep, 

                                        epochs=epoches*1, 

                                        verbose=1, callbacks=None, 

                                        validation_data=validBatch[i-1],

                                        validation_steps=vs,  

                                        workers=1) for i in range (1,5)]

    timin()





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

submission["EncodedPixels"] = ""



i =0 



def mask2rle(img):

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def getResults(x, image,models):

    inputImg = np.array(cv2.resize(cv2.imread(testImgPath+image),(800,128))).reshape(128, 800, 3).reshape(1, 128, 800, 3)

    prediction = models[x-1].predict(inputImg,batch_size=None, verbose=0, steps=None)

    if x==1:

       return  np.where(cv2.blur(cv2.blur(prediction[0,:,:,0],(5,5)),(5,5))>0.995,1,0)

    elif x==2:

       return  np.where(cv2.blur(cv2.blur(prediction[0,:,:,0],(5,5)),(5,5))>0.995,1,0)

    elif x==3:

       return  np.where(cv2.blur(cv2.blur(prediction[0,:,:,0],(5,5)),(5,5))>0.97,1,0)

    elif x==4:

       return  np.where(cv2.blur(cv2.blur(prediction[0,:,:,0],(5,5)),(5,5))>0.97,1,0)





def getResult(image):

    models = [model1,model2,model3,model4]

    return [getResults( x, image, models) for x in labelsInt]

    

if(True):

    models = [model1,model2,model3,model4]

    for image in k:

        results =  getResult(image)

        i = i +1

        for l,r in zip(labels, results):

            if (len(r))>0:

                submission.loc[submission['ImageId_ClassId']==image+"_"+l,["EncodedPixels"]] =  mask2rle(r).strip()

        if (i%50==0):

            print("done",i,timin())

    submission.to_csv("submission.csv",index=False)

if (False):

    models = [model1,model2,model3,model4]

    x, y = next(SafePandasDG(df1, None, 10, False, True))   

    print(x[0].shape)

    for ii in range(0, x.shape[0] ):

        plt.figure(figsize=(35,10))

        plt.imshow(x[ii], cmap = 'Greys', interpolation = 'bicubic')

        for i in range(0,4):

            predict = models[i].predict(x,batch_size=10, verbose=0, steps=None)

            plt.figure(figsize=(35,10))

            plt.imshow(np.where(cv2.blur(cv2.blur(predict[ii,:,:,0],(5,5)),(5,5))>0.26,1,0), cmap = 'Greys', interpolation = 'bicubic')

clss = [1,2,3,4]

modelss = [model1,model2,model3,model4]

for m in modelss:

    for c in clss:

        print("Evaluate for model",c, m)

        print(m.evaluate_generator(getBatch[c-1], steps=16,   verbose=0))
