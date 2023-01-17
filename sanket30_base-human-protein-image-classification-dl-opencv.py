import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import gc
from keras.preprocessing.image import ImageDataGenerator

#================================
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

#================================

import matplotlib
#matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
#from imutils import paths
import numpy as np
#import argparse
import random
import pickle
import cv2
import os
filepath0="../input/train/"
filepath1="../input/test/"
train_df=pd.read_csv("../input/train.csv")
train_image=os.listdir("../input/train/")
greenimage= [n for n in train_image if "green" in n]
gdf=pd.DataFrame({"imagename":greenimage})

gdf.shape
dff=pd.concat([train_df,gdf],axis=1)
dff.head(3)
df=dff[0:1000]
df.shape
img_height=512
img_width=512
image=[]
#labels = []
for i in df['imagename']:
        images = cv2.imread(filepath0+i,0) 
        images = cv2.resize(images, (img_width, img_height))
        image.append(images)
       
len(image)
plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(image[i])
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
   
plt.show()
labels = []
for i in df['Target']:
    li = list(i.split(" ")) 
    labels.append(li)
len(labels)
labels[0:5]
image = np.array(image)
labels = np.array(labels)
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
 
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
gc.collect()
(trainX, testX, trainY, testY) = train_test_split(image,labels, test_size=0.2, random_state=42)

trainX = trainX.reshape(trainX.shape[0], img_width, img_height,1) 
testX = testX.reshape(testX.shape[0], img_width, img_height,1) 
trainY.shape
aug = ImageDataGenerator()
EPOCHS = 20
INIT_LR = 1e-3
BS = 32

height=512
width=512
depth=1
chanDim = -1
classes=28, 
finalAct="sigmoid"


inputShape = (height, width, depth)

model = Sequential()
# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same",
input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
model.add(Dense(27))
model.add(Activation(finalAct))
 
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=1),validation_data=(testX, testY),steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)

#H=model.fit(trainX, trainY, batch_size=BS,validation_data=(testX, testY),steps_per_epoch=len(trainX) // BS,epochs=EPOCHS)
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validaiton'], loc='upper left')
plt.show()
sub_df=pd.read_csv("../input/sample_submission.csv")
test_image=os.listdir("../input/test/")
testgreenimage= [n for n in test_image if "green" in n]
#testgdf=pd.DataFrame({"imagename":testgreenimage})

sub_df.shape
len(testgreenimage)
X_test=[]
Y_test=[]

for i in testgreenimage[20:21]:
    image = cv2.imread(filepath1+i,0) 
    images = cv2.resize(image, (img_width, img_height))
    X_test.append(images)
    Y_test.append(images)
X_test=np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height,1) 
X_test.shape
proba = model.predict(X_test)[0]
proba.shape
idxs = np.argsort(proba)[::-1][:2]
# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):

    label = "{}: {:}%".format(mlb.classes_[j], proba[j] * 100)

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
    print("{}: {:}%".format(label, int(p * 100)))


