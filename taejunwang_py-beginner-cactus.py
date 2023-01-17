# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from glob import glob



from PIL import Image

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tqdm import tqdm

from keras import layers 

from keras import models

from keras import optimizers



%matplotlib inline
os.listdir()
import shutil

shutil.copytree('../input', './input')
os.listdir('input')
# Pathname set

input_dir = 'input/'

train_dir = 'input/train/train/' #train image check!

test_dir = 'input/test/test/'    #test image check!

train_csv_dir = input_dir + 'train.csv'
# CSV file check: train.csv



df_tr = pd.read_csv(input_dir+'train.csv', sep=",", quotechar="'")

print(">> df_tr_shape:", df_tr.shape)

print(">>", len(df_tr), "rows", "\n")

print(df_tr[:16])
# CSV file check: sample_submission.csv



df_te = pd.read_csv(input_dir + 'sample_submission.csv', sep=",", quotechar="'")

print(">> df_te_shape:", df_te.shape)

print(">>", len(df_te), "rows", "\n")

print(df_te[:16])
# Filename list



Fname_list = df_tr['id']

Fname_list[:16]
# Label list



Lname_list = df_tr['has_cactus']

Lname_list[:16]
# Label count



print(">> Total", len(Lname_list), "files\n")

print(Lname_list.value_counts(), "\n")



Table = {}

for i in range(0, len(df_tr)):

    row = df_tr.iloc[i,:]

    Table[row.id] = row.has_cactus

df_tr.head(16)
# Train image check



plt.figure(figsize=[15,10])



for i in range(0, 15):

    im_train = np.array(Image.open(train_dir + Fname_list[i]))

    #im_train = np.array(Image.open(Fname_list[i]))

    plt.subplot(3, 5, i+1)

    plt.imshow(im_train)

    plt.grid(True)

    plt.title("#" + str(i) + " - [" + str(Lname_list[i]) + "]")

    i = i+1

plt.show()
train_csv =  pd.read_csv(train_csv_dir)

train_csv
images_tr = []

labels_tr = []



image_train_name = Fname_list.values

lable_train_name = Lname_list.values



for i in tqdm(image_train_name):

    images_tr.append(cv2.imread(train_dir + i))

    labels_tr.append(train_csv[Fname_list == i]['has_cactus'].values[0]) 
images_tr = np.asarray(images_tr)

images_tr = images_tr.astype('float32')

images_tr = images_tr / 255

labels_tr = np.asarray(labels_tr)
# Generate a model



model = models.Sequential()



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2))) 

 

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Flatten()) 

model.add(layers.Dense(512, activation='relu')) 

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
# Parameters



validation_split = .2

batch_size = 150

epochs_num = 5

learning_rate = .001
# Data training



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=learning_rate), metrics=['acc'])



hist = model.fit(images_tr, labels_tr, batch_size=batch_size, epochs=epochs_num, validation_split=validation_split)
hist.history.keys()
plt.figure(figsize=(15,4))



# Accuracy history

plt.subplot(121)

plt.plot(hist.history['acc'], c='blue')

plt.plot(hist.history['val_acc'], c='green')

plt.title('Train model: Accuracy', fontsize=15)

plt.ylabel('Accuracy', fontsize=15)

plt.xlabel('Epoch_num', fontsize=15)

plt.grid(True)

plt.legend(['train', 'test'], loc=4)



# Loss history

plt.subplot(122)

plt.plot(hist.history['loss'], c='blue')

plt.plot(hist.history['val_loss'], c='green')

plt.title('Train model: Loss', fontsize=15)

plt.ylabel('Loss', fontsize=15)

plt.xlabel('Epoch_num', fontsize=15)

plt.grid(True)

plt.legend(['train', 'test'], loc=1)

plt.show()
# Test data check



test_bin = []

Test_imgs = []



for i in tqdm(os.listdir(test_dir)):

    test_bin.append(cv2.imread(test_dir + i))     

    Test_imgs.append(i)

    

test_bin = np.asarray(test_bin)

test_bin = test_bin.astype('float32')

test_bin = test_bin / 255
# Predictions



test_pred = model.predict(test_bin)

print(">> test_pred_shape:", test_pred.shape)

print(">> Predictions:", test_pred[-1][0]*100, "%")

# Initialization: Input folder

shutil.rmtree('input')