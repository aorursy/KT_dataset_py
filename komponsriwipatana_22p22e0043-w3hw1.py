# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, \
ReLU, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras import Model
import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import copy
# Full pre-activation
def res_block(x_in,n,s=1,k=3):
  x = BatchNormalization()(x_in)
  x = ReLU()(x)
  x = Conv2D(n,k,s,padding="same")(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(n, k, 1, padding="same")(x)
  if s==2:
    x_in = Conv2D(n, 1, 2, padding="same")(x_in)
  x = Add()([x_in, x])
  return x
def resnet34(input_shape=(224, 224, 3), classes=1000):
  x_in = Input(shape=input_shape)

  x = Conv2D(64, 7, 2, padding="same")(x_in)
  x = MaxPooling2D(3, 2, padding="same")(x)

  for i in range(3):
    x = res_block(x, 64)

  x = res_block(x, 128, 2)
  for i in range(3):
    x = res_block(x, 128)

  x = res_block(x, 256, 2)
  for i in range(5):
    x = res_block(x, 256)

  x = res_block(x, 512, 2)
  for i in range(2):
    x = res_block(x, 512)

  x = GlobalAveragePooling2D()(x)
  x = Dense(classes, activation='softmax')(x)

  model = Model(x_in, x)
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )
  return model
def load_image( infilename , imgW=400,imgH=400 ) :
    img = Image.open( infilename )
    img.load()
    new_image = img.resize((imgW, imgH))
    
    data = np.asarray( new_image, dtype="float32" )
        
    return data
def getDataSet(imgDir,imgPaths,labels,imgW=224,imgH=224):    
    xTrain = []
    yTrain = []    
    
    tmppaths=[]
    
    tmppaths = imgPaths[:]
    yTrain = labels[:]
    
    for i in tmppaths:
        filename = imgDir + i
        imgArray = load_image(filename,imgW=img_width,imgH=img_height)
        xTrain.append(imgArray)
        
    return xTrain,yTrain
def getValDataSet(imgDir,imgPaths,imgW=224,imgH=224):    
    xTrain = []
        
    tmppaths=[]
    
    tmppaths = imgPaths[:]
        
    for i in tmppaths:
        filename = imgDir + i
        imgArray = load_image(filename,imgW=img_width,imgH=img_height)
        xTrain.append(imgArray)
        
    return xTrain
def getDataSetByBatchIndex(imgDir,imgPaths,labels,batchsize=10,batchindex=0,imgW=224,imgH=224):    
    xTrain = []
    yTrain = []    
    
    tmppaths=[]
    
    if ((batchindex+1)*(batchsize)) > len(imgPaths):
        tmppaths = imgPaths[(batchindex)*(batchsize):]
        yTrain = labels[(batchindex)*(batchsize):]
    else:
        tmppaths = imgPaths[(batchindex)*(batchsize):(batchindex+1)*(batchsize)]
        yTrain = labels[(batchindex)*(batchsize):(batchindex+1)*(batchsize)]
    
    for i in tmppaths:
        filename = imgDir + i
        imgArray = load_image(filename,imgW=img_width,imgH=img_height)
        xTrain.append(imgArray)
        
    return xTrain,yTrain    
img_width = 100
img_height = 100

nclass = 2

image_dir = '/kaggle/input/superai-w3hw1/train/train/images/'

dataCSV = pd.read_csv('/kaggle/input/superai-w3hw1/train/train/train.csv')
dataCSV = dataCSV.sample(frac = 1)

image_paths, labels = dataCSV['id'],dataCSV['category']

xDS , yDS = getDataSet(image_dir,image_paths,labels,img_width,img_height)
mask =int(len(yDS)*0.8)
Xtrain = np.array(xDS[:mask])
Ytrain = np.array(yDS[:mask])

Xtest = np.array(xDS[mask:])
Ytest = np.array(yDS[mask:])

Xtrain.shape
# Ytrain.shape
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(Xtrain)

number_of_batches = 10
batchsize=20

batches = 0

X_new = []
Y_new = []
for X_batch, Y_batch in datagen.flow(Xtrain, Ytrain, batch_size=batchsize):
    X_new.append(X_batch)
    Y_new.append(Y_batch) 

    batches += 1
    if batches >= number_of_batches:
        break

tmpXNew = []
tmpYNew = []

Xtrain = list(Xtrain)
Ytrain = list(Ytrain)

for i in range(number_of_batches):
    for j in range(batchsize):
        Xtrain.append(X_new[i][j])
        Ytrain.append(Y_new[i][j])


# Xtrain = np.append(Xtrain, tmpXNew)

print(len(Xtrain))
# Xtrain[1579].shape

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

resnet = resnet34((img_width, img_height, 3), nclass)
resnet = keras.models.load_model("./resnet")
fitHistory = resnet.fit(Xtrain, Ytrain, epochs=50, validation_data=(Xtest, Ytest))
resnet.save('./resnet')
print(fitHistory.history.keys())
# summarize history for accuracy
plt.plot(fitHistory.history['accuracy'])
plt.plot(fitHistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(fitHistory.history['loss'])
plt.plot(fitHistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
resnet = keras.models.load_model("./resnet")
image_val = '/kaggle/input/superai-w3hw1/val/val/images/'

paths2=[]
labels2=[]
for dirname, _, filenames in os.walk(image_val):
    for filename in filenames:
        paths2.append(filename)

xDS2 = getValDataSet(image_val,paths2,img_width,img_height)

print(len(paths2))
print(len(xDS2))
xData = np.array(xDS2[:])
xData.shape

predi = resnet.predict(xData)
labels2 = predi.argmax(axis=1)

df = pd.DataFrame() 
df['id']=paths2
df['category']=labels2

df.to_csv('./val.csv',header=True, index = False)

