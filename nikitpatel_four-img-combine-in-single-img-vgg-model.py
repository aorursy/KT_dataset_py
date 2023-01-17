# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter

import os
print(os.listdir("../input"))
#import training data
train = pd.read_csv("../input/train.csv")
print(train.head())

#map of targets in a dictionary
subcell_locs = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Peroxisomes",
9:  "Endosomes",
10:  "Lysosomes",
11:  "Intermediate filaments",   
12:  "Actin filaments",
13:  "Focal adhesion sites",   
14:  "Microtubules",
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle",
18:  "Microtubule organizing center",  
19:  "Centrosome",
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions", 
23:  "Mitochondria",
24:  "Aggresome",
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
print("The image with ID == 1 has the following labels:", train.loc[1, "Target"])
print("These labels correspond to:")
for location in train.loc[1, "Target"].split():
    print("-", subcell_locs[int(location)])

#reset seaborn style
sns.reset_orig()

#get image id
im_id = train.loc[1, "Id"]

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)

#get each image channel as a greyscale image (second argument 0 in imread)
green = cv2.imread('../input/train/{}_green.png'.format(im_id), 0)
red = cv2.imread('../input/train/{}_red.png'.format(im_id), 0)
blue = cv2.imread('../input/train/{}_blue.png'.format(im_id), 0)
yellow = cv2.imread('../input/train/{}_yellow.png'.format(im_id), 0)

#display each channel separately
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(15, 15))
ax[0, 0].imshow(green, cmap="greens")
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(red, cmap="reds")
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(blue, cmap="blues")
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(yellow, cmap="yellows")
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()
labels_num = [value.split() for value in train['Target']]
labels_num_flat = list(map(int, [item for sublist in labels_num for item in sublist]))
labels = ["" for _ in range(len(labels_num_flat))]
for i in range(len(labels_num_flat)):
    labels[i] = subcell_locs[labels_num_flat[i]]

fig, ax = plt.subplots(figsize=(15, 5))
pd.Series(labels).value_counts().plot('bar', fontsize=14)
train_img = os.listdir("../input/train/")
test_img = os.listdir("../input/test/")

train_path = "../input/train/"
test_path = "../input/test/"

train_df = pd.DataFrame(train_img,columns=['image_id'])
test_df = pd.DataFrame(test_img,columns=['image_id'])
print("Number of Total Train Image : ",len(train_df))
print("Number of Test Train Image : ",len(test_df))
color = []
for n in train_img:
    if "red" in n:
       color.append('red')
    elif "blue" in n:
       color.append('blue')
    elif "yellow" in n:
       color.append('yellow')
    elif "green" in n:
       color.append('green')
train_df['c_name'] = pd.DataFrame(color)
color = []
for n in test_img:
    if "red" in n:
       color.append('red')
    elif "blue" in n:
       color.append('blue')
    elif "yellow" in n:
       color.append('yellow')
    elif "green" in n:
       color.append('green')  
test_df['c_name'] = pd.DataFrame(color)
plt.figure(figsize=(15,12))
train_df["c_name"].value_counts().plot(kind="bar")
plt.xlabel("Counts")
plt.ylabel("Colors")
plt.legend("Colors")
plt.title("Color Image Counts")
train_df.head(5)
test_df.head(5)
train_df['id'] = train_df['image_id'].str.split('_').str[0]
test_df['id'] = test_df['image_id'].str.split('_').str[0]
print("Total Number of Unique Image on Train Data ",len(train_df['id'].value_counts()))
print("Total Number of Unique Image on Test Data ",len(test_df['id'].value_counts()))
train_df = train_df.sort_values(by=['id', 'c_name']).reset_index(drop=True)
test_df = test_df.sort_values(by=['id', 'c_name']).reset_index(drop=True)
train_df.head(12)
test_df.head(12)
import cv2
import gc
gc.collect()
import matplotlib.pyplot as plt
img_1 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',0)
img_2 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',0)
img_3 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',0)
img_4 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png',0)
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(20, 20))
ax[0, 0].imshow(img_1)
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(img_2)
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(img_3)
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(img_4)
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()
img_1 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',0)
img_2 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',0)
img_3 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',0)
img_4 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png',0)
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(20, 20))
ax[0, 0].imshow(img_1, cmap="blues")
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(img_2, cmap="greens")
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(img_3, cmap="reds")
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(img_4, cmap="yellows")
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()
img_1 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',0)
img_2 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',0)
img_3 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',0)
img_4 = cv2.imread('../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png',0)

no_img = 4
img = img_1/no_img + img_2/no_img + img_3/no_img + img_4/no_img
img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
train_df1 = train_df[0:12]

for name, group in train_df1.groupby(['id'])['image_id']:
    img_1 = cv2.imread('../input/train/'+group.values[0],0)
    img_2 = cv2.imread('../input/train/'+group.values[1],0)
    img_3 = cv2.imread('../input/train/'+group.values[2],0)
    img_4 = cv2.imread('../input/train/'+group.values[3],0)

    no_img = 4
    img = img_1/no_img + img_2/no_img + img_3/no_img + img_4/no_img
    print(img.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()
train_image = []
train = train_df[0:6000]
no_img = 4
for name, group in train.groupby(['id'])['image_id']:
    img_1 = cv2.imread('../input/train/'+group.values[0],0)
    img_2 = cv2.imread('../input/train/'+group.values[1],0)
    img_3 = cv2.imread('../input/train/'+group.values[2],0)
    img_4 = cv2.imread('../input/train/'+group.values[3],0)
    img = []
    img = img_1/no_img + img_2/no_img + img_3/no_img + img_4/no_img
    train_image.append(img)
train = pd.read_csv("../input/train.csv")
train = train.sort_values(['Id']).reset_index(drop=True)
train.head(5)
labels = []
for i in train['Target'][0:1500]:
    li = list(i.split(" ")) 
    labels.append(li)
print("length of Traget Variable :",len(labels))
image = np.array(train_image)
labels = np.array(labels)
gc.collect()
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
 
#loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
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
img_width = 512
img_height = 512
(trainX, testX, trainY, testY) = train_test_split(image,labels, test_size=0.3, random_state=42)

trainX = trainX.reshape(trainX.shape[0], img_width, img_height,1) 
testX = testX.reshape(testX.shape[0], img_width, img_height,1) 
aug = ImageDataGenerator()
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
depth=1
chanDim = -1
classes=28, 
finalAct="sigmoid"


inputShape = (img_width, img_height, depth)

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
model_vgg = model.fit_generator(aug.flow(trainX, trainY, batch_size=1),validation_data=(testX, testY),steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1)

fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(model_vgg.epoch, model_vgg.history["loss"], label="Train loss")
ax[0].plot(model_vgg.epoch, model_vgg.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(model_vgg.epoch, model_vgg.history["acc"], label="Train acc")
ax[1].plot(model_vgg.epoch, model_vgg.history["val_acc"], label="Validation acc")
ax[0].legend()
ax[1].legend()

#sub = pd.read_csv("../input/sample_submission.csv")
# test_image = []
# for name, group in test_df.groupby(['id'])['image_id']:
#     img_1 = cv2.imread('../input/test/'+group.values[0],0)
#     img_2 = cv2.imread('../input/test/'+group.values[1],0)
#     img_3 = cv2.imread('../input/test/'+group.values[2],0)
#     img_4 = cv2.imread('../input/test/'+group.values[3],0)
#     img = []
#     img = img_1/no_img + img_2/no_img + img_3/no_img + img_4/no_img
#     i = i + 1
#     print(i)
#     test_image.append(img)
