

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,sys
import matplotlib.pyplot as plt
import cv2
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

%matplotlib inline
# Any results you write to the current directory are saved as output.
def fetchImage(imagePath,imageShape):
    frame = cv2.imread(imagePath,cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(imageShape[0],imageShape[1]))
    return frame

def shuffle(lst):
    indicies = np.random.permutation(np.arange(len(lst[0])))
    shuffled = []
    for item in lst:
        item = np.array(item)[indicies]
        shuffled.append(item)
    return shuffled
    
home = "../input"
trainHome = home+"/train"
testHome = home+"/test"
imageShape = (64,64,3)
classNames = os.listdir(trainHome)
classNames_to_int = dict((className,i) for i,className in enumerate(classNames))
int_to_className = dict((i,className) for className,i in classNames_to_int.items())
num_classes = len(classNames_to_int)

print("Class count:",num_classes)
print("Class names:",classNames)


img_class = []
class_img_freq = dict()
for className in classNames:
    classImageNames = os.listdir(trainHome+"/"+className)
    class_img_freq[className] = len(classImageNames)
    for imgName in classImageNames:
        img_class.append([imgName,className])
img_class = np.array(img_class)
print("Total training images",img_class.shape)

labels,freqs = zip(*list(class_img_freq.items()))
plt.bar(labels,freqs)
plt.xticks(range(len(labels)),labels,rotation="vertical")
plt.xlabel("Plant Species")
plt.ylabel("Frequency")
plt.show()
# Load training images.
X = []
Y = []
temp = 0
for imgName,imgClass in img_class:
    temp += 1
    img = fetchImage(trainHome+"/"+imgClass+"/"+imgName,imageShape)
    X.append(img)
    Y.append(classNames_to_int[imgClass])
    if temp%100 == 0:
        sys.stdout.write("\rLoaded {}/{} images.".format(temp,img_class.shape[0]))
print("Loaded {}/{} images.".format(temp,img_class.shape[0]))
    
X = np.array(X).astype(np.float32)/255
Y = np.array(Y)
X,Y = shuffle([X,Y])
Y = keras.utils.to_categorical(Y,num_classes = num_classes)
print("X.shape",X.shape)
print("Y.shape",Y.shape)
# Create model.
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from keras.layers import Dropout,Dense,Flatten,Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop,SGD
from keras.layers import LeakyReLU
from keras.regularizers import l2
import keras.backend as K

activation = LeakyReLU(alpha=0.1)
conv_filter_size = (2,2)
if K.image_dim_ordering() == "th":
        concat_axis = 1
elif K.image_dim_ordering() == "tf":
        concat_axis = -1

model = Sequential()
model.add(Conv2D(64,conv_filter_size,padding="same",input_shape=imageShape,use_bias=False,kernel_regularizer=l2(1e-4)))
model.add(activation)
model.add(Conv2D(64,conv_filter_size,padding="same",use_bias=False,kernel_regularizer=l2(1e-4)))
model.add(activation)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
model.add(Dropout(0.))

model.add(BatchNormalization(axis=concat_axis,beta_regularizer=l2(1e-4),gamma_regularizer=l2(1e-4)))
model.add(Conv2D(128,conv_filter_size,padding="same",use_bias=False,kernel_regularizer=l2(1e-4)))
#model.add(Activation(activation))
model.add(activation)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
model.add(Dropout(0.))

model.add(BatchNormalization(axis=concat_axis,beta_regularizer=l2(1e-4),gamma_regularizer=l2(1e-4)))
model.add(Conv2D(512,conv_filter_size,padding="same",use_bias=False,kernel_regularizer=l2(1e-4)))
#model.add(Activation(activation))
model.add(activation)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
model.add(Dropout(0.4))

model.add(BatchNormalization(axis=concat_axis,beta_regularizer=l2(1e-4),gamma_regularizer=l2(1e-4)))
model.add(Conv2D(1024,conv_filter_size,padding="same",use_bias=False,kernel_regularizer=l2(1e-4)))
#model.add(Activation(activation))
model.add(activation)
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
model.add(Dropout(0.4))

model.add(GlobalAveragePooling2D())

#model.add(BatchNormalization(axis=concat_axis,beta_regularizer=l2(1e-4),gamma_regularizer=l2(1e-4)))
#model.add(Dense(1024))
##model.add(Activation(activation))
#model.add(activation)
#model.add(Dropout(0.4))

#model.add(BatchNormalization(axis=concat_axis,beta_regularizer=l2(1e-4),gamma_regularizer=l2(1e-4)))
#model.add(Dense(1024))
##model.add(Activation(activation))
#model.add(activation)
#model.add(Dropout(0.4))

model.add(Dense(num_classes,activation="softmax"))

model.summary()

model.compile(optimizer=RMSprop(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
epochs = 20
batch_size = 16
checkpoint = ModelCheckpoint("PlantSeedling.hdf5",verbose=1)
training_type = "non_generator" # {'generator','non_generator'}
# GENERATOR training.
from keras.preprocessing.image import ImageDataGenerator

if training_type == 'generator':
    dataGen = ImageDataGenerator(shear_range=0.1,zoom_range=0.1,horizontal_flip=True)
    valLen = int(X.shape[0]*0.2)
    X_train = X[valLen:]
    Y_train = Y[valLen:]
    X_val = X[:valLen]
    Y_val = Y[:valLen]

    num_batches = int(X_train.shape[0]/batch_size)
    hist = model.fit_generator(dataGen.flow(X_train,Y_train,batch_size=batch_size),epochs=50,
                               steps_per_epoch=num_batches,
                              validation_data=(X_val,Y_val))
# NON-GENERATOR training.
if training_type == 'non_generator':
    hist = model.fit(X,Y,batch_size = batch_size,epochs = 50,
                    validation_split=0.2)
plt.plot(hist.history["loss"],label="loss")
plt.plot(hist.history["val_loss"],label="val_loss")
plt.legend(loc="upper left")
plt.show()

plt.plot(hist.history["acc"],label="acc")
plt.plot(hist.history["val_acc"],label="val_acc")
plt.legend(loc="upper left")
plt.show()
# Load Prediction Data.
testImgNames = sorted(os.listdir(testHome))
X_test = [fetchImage(testHome+"/"+testImgName,imageShape) for testImgName in testImgNames]
X_test = np.array(X_test).astype(np.float32)/255
print("X_test.shape",X_test.shape)

# Prediction
Y_pred = model.predict(X_test)
intClasses = np.argmax(Y_pred,axis=1)

strClasses = [int_to_className[intClass] for intClass in intClasses]
#strClasses
result = list(zip(testImgNames,strClasses))
for imgName,strClass in result:
    print(imgName+","+strClass)
