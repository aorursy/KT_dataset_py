# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras import optimizers, Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.utils import to_categorical

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline
train_path = '/kaggle/input/intel-image-classification/seg_train/seg_train/'

test_path = '/kaggle/input/intel-image-classification/seg_test/seg_test/'

pred_path = '/kaggle/input/intel-image-classification/seg_pred/seg_pred/'
pixel = 50
train_dir = os.listdir(train_path)

print(train_dir)
test_dir = os.listdir(test_path)

print(test_dir)
pred_dir = os.listdir(pred_path)

print(len(pred_dir))
train_dir_street = train_path+train_dir[0]+'/'

train_dir_buildings = train_path+train_dir[1]+'/'

train_dir_mountain = train_path+train_dir[2]+'/'

train_dir_sea = train_path+train_dir[3]+'/'

train_dir_forest = train_path+train_dir[4]+'/'

train_dir_glacier = train_path+train_dir[5]+'/'



training_dir_list = [train_dir_street, train_dir_buildings, train_dir_mountain, train_dir_sea, train_dir_forest, train_dir_glacier]



test_dir_street = test_path+test_dir[0]+'/'

test_dir_buildings = test_path+test_dir[1]+'/'

test_dir_mountain = test_path+test_dir[2]+'/'

test_dir_sea = test_path+test_dir[3]+'/'

test_dir_forest = test_path+test_dir[4]+'/'

test_dir_glacier = test_path+test_dir[5]+'/'



testing_dir_list = [test_dir_street, test_dir_buildings, test_dir_mountain, test_dir_sea, test_dir_forest, test_dir_glacier]
test_img = cv2.imread('/kaggle/input/intel-image-classification/seg_train/seg_train/street/1586.jpg')

print(test_img.shape)
train = []

for i in range(len(training_dir_list)):

    images = os.listdir(training_dir_list[i])

    length = len(images)

    for j in tqdm(range(length)):

        path = training_dir_list[i]+images[j]

        pic = cv2.imread(path)

        pic = cv2.resize(pic,(pixel,pixel))

        pic = pic/255

        train.append([pic,i])

train = np.array(train)        
test = []

for i in range(len(testing_dir_list)):

    images = os.listdir(testing_dir_list[i])

    length = len(images)

    for j in tqdm(range(length)):

        path = testing_dir_list[i]+images[j]

        pic = cv2.imread(path)

        pic = cv2.resize(pic,(pixel,pixel))

        pic = pic/255

        test.append([pic,i])

test = np.array(test)        
for i in range(5):

    np.random.shuffle(train)

    np.random.shuffle(test)
X_train = []

Y_train = []

for i in train:

    X_train.append(i[0])

    Y_train.append(i[1])

X_train = np.array(X_train)

Y_train = np.array(Y_train)

Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
X_test = []

Y_test = []

for i in test:

    X_test.append(i[0])

    Y_test.append(i[1])

X_test = np.array(X_test)

Y_test = np.array(Y_test)

Y_test = np.reshape(Y_test,(Y_test.shape[0],1))
Y_train = to_categorical(Y_train)

Y_test = to_categorical(Y_test)
print(X_train.shape,'\t',Y_train.shape)

print(X_test.shape,'\t',Y_test.shape)
# Fitting CNN



model = Sequential()

model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(pixel,pixel,3)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32, kernel_size=4, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Flatten())

model.add(Dense(6, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,validation_split = 0.33, epochs=15, verbose=1)
model.evaluate(X_test, Y_test)
plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.plot(history.history['val_accuracy'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Values for Accuracy and Loss')

plt.legend(['Training Accuracy','Training Loss','Validation Accuracy','Validation Loss'])
model.summary()
prediction = []

real = []

for i in tqdm(range(len(pred_dir))):

    path = pred_path+pred_dir[i]

    pic = cv2.imread(path)

    pic = pic/255

    real.append(pic)

    pic = cv2.resize(pic,(pixel,pixel))

    prediction.append(pic)

prediction = np.array(prediction)
print(prediction.shape)
target = ['Street','Building','Mountain','Sea','Forest','Glacier']
pr = model.predict_classes(prediction)
val = 990

print(target[pr[val]])

plt.imshow(real[val])