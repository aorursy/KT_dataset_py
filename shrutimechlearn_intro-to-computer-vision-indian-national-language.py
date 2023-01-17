import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 

import copy

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import glob

import os

from tqdm import tqdm 

import matplotlib.image as mpimg

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
dataset = pd.read_csv("/kaggle/input/devanagari-character-set/data.csv")
dataset.head()
dataset.shape
def read_data():



    data_set_full = [] 

    for folder_path in glob.glob('/kaggle/input/devanagari-character-set/Images/Images/*'):



        class_name = folder_path.split('/')[-1]

        print(class_name)



        for img in tqdm(os.listdir(folder_path)): 

            path = os.path.join(folder_path, img)

            img = cv2.imread(path)

            data_set_full.append([np.array(img), class_name])

            #print(data_set_full)

            #break

        

    return data_set_full
## Commented due to long run time once after the data had been generated.

##full_data = read_data()
##len(full_data)
plt.figure(figsize = (10,20))

_ = sns.countplot(x=None, y=dataset.character)
dataset.drop(['character'], axis =1).iloc[0].max()
img = (dataset.drop(['character'],axis=1).iloc[0]).to_numpy().reshape(32,32)

_ = plt.imshow(img, cmap = 'gray')
#import Image



def is_grey_scale(img_path):

    img = Image.open(img_path).convert('RGB')

    w,h = img.size

    for i in range(w):

        for j in range(h):

            r,g,b = img.getpixel((i,j))

            if r != g != b: return False

    return True





### Basically, check every pixel to see if it is grayscale (R == G == B)
ones_array = np.ones([100, 100, 3], dtype=np.uint8)

_ = plt.imshow(ones_array)
red_array = copy.deepcopy(ones_array)

red_array[:,:,0] = 255

red_array[:,:,1] = 0

red_array[:,:,2] = 0



_ = plt.imshow(red_array)
any_gray_array = copy.deepcopy(ones_array)

any_gray_array[:,:,0] = 200

any_gray_array[:,:,1] = 200

any_gray_array[:,:,2] = 200



_ = plt.imshow(any_gray_array)
any_gray_array = copy.deepcopy(ones_array)

any_gray_array[:,:,0] = 150

any_gray_array[:,:,1] = 150

any_gray_array[:,:,2] = 150



_ = plt.imshow(any_gray_array)
any_gray_array = copy.deepcopy(ones_array)

any_gray_array[:,:,0] = 100

any_gray_array[:,:,1] = 100

any_gray_array[:,:,2] = 100



_ = plt.imshow(any_gray_array)
x = dataset.drop(['character'],axis = 1)

y_text = dataset.character
x.shape
from sklearn.preprocessing import LabelBinarizer

binencoder = LabelBinarizer()

y = binencoder.fit_transform(y_text)
y[0]
x = x.values.reshape(x.shape[0],32,32,1)
x.shape
print(x.max())

print(x.mean())

print(x.sum())
x = x/255.0
print(x.max())

print(x.mean())

print(x.sum())
x.shape
y.shape
(unique, counts) = np.unique(y, return_counts=True, axis = 0)

frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T
frequencies
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42, stratify = y)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
(unique, counts) = np.unique(y_train, return_counts=True, axis = 0)

frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T
frequencies
(unique, counts) = np.unique(y_test, return_counts=True, axis = 0)

frequencies = np.asarray((binencoder.inverse_transform(unique), counts)).T
frequencies
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (32,32,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(46, activation = "softmax"))
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 7

batch_size = 86
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 

       validation_data = (x_test, y_test), verbose = 2)
scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)