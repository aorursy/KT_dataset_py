# Helper libraries

import os

import numpy as np

import cv2

from PIL import Image

import json

from matplotlib import pyplot as plt

import pandas as pd



# TensorFlow and Keras

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
path = '../input/noisy-and-rotated-scanned-documents/scan_doc_rotation/'
# Load and open images

names = [ file for file in os.listdir(path+'images/') ]

names = sorted(names)

N = len(names)



print(names[:10])



# Load two images to check

images = [Image.open(path+'images/'+names[i]) for i in range(2)]



plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(images[0], cmap = 'gray')

plt.subplot(122),plt.imshow(images[1], cmap = 'gray')



print('Total number of images: %d'%N)
eg_img = path+'/images/'+names[18]

img = cv2.imread(eg_img)

img = img[:,:,0] #zeroth component is the red from RGB channel ordering

f = cv2.dft(np.float32(img))

fshift = np.fft.fftshift(f)

f_abs = np.abs(fshift) + 1.0 #shift to ensure no zeroes are present in image array

f_img = 20 * np.log(f_abs)



plt.figure(figsize=(14,8))



plt.subplot(121),plt.imshow(img, 'gray')

plt.title('Original scan')



plt.subplot(122),plt.imshow(f_img, 'gray')

plt.title('FFT of scan')

plt.colorbar() 

plt.show()
from skimage.restoration import denoise_tv_chambolle



img = denoise_tv_chambolle(img, weight=1.0, multichannel=0)

                               

f = cv2.dft(np.float32(img))

fshift = np.fft.fftshift(f)

f_abs = np.abs(fshift) + 1.0

f_img = 20 * np.log(f_abs)



plt.figure(figsize=(14,8))

plt.subplot(121),plt.imshow(img, 'gray')

plt.subplot(122),plt.imshow(f_img, 'gray')

os.system('mkdir -p ./processed')

cv2.imwrite(path+'/processed/'+names[18], f_img)

plt.colorbar() 

plt.show()
fft_images = []

for i in range(N):

    img = cv2.imread(path+'/images/'+names[i])

    img = img[:,:,0]

    img = denoise_tv_chambolle(img, weight=1.0, multichannel=0)

    f = cv2.dft(np.float32(img))

    fshift = np.fft.fftshift(f)

    f_abs = np.abs(fshift) + 1.0 # shift to avoid np.log(0) = nan

    f_img = 20 * np.log(f_abs)

    fft_images.append( f_img )

    cv2.imwrite(path+'/processed/'+names[i], f_img)
# Load and open labels

label_names = [ file for file in os.listdir(path+'./labels') ]

label_names = sorted(label_names)

M = len(label_names)



print(label_names[:10])



labels = [ np.loadtxt(path+'./labels/'+label_names[j])

for j in range(M) ]

labels = [ round(float(labels[j])) for j in range(M) ]



# Load first 10 labels

[print(labels[i]) for i in range(10)]

print('Total number of labels %d'%len(labels))
# Deserialize JSON data lists for training and test sets

with open(path+'train_list.json') as train_data:

    train = json.load(train_data)

    

train_size = len(train)

print('Training set size: %d'%train_size)



with open(path+'test_list.json') as test_data:

    test = json.load(test_data)

    

test_size = len(test)

print('Test set size: %d'%test_size)
# Get images into tensor form

image_arr = [ tf.keras.preprocessing.image.img_to_array(fft_images[i]) 

for i in range(N) ]



# get pixel dimensions of image

img_height = image_arr[0].shape[0]

img_width = image_arr[0].shape[1]



# Training and test image stacks

X_train = tf.stack(image_arr[:train_size], axis=0, name='train_set')

X_test = tf.stack(image_arr[-test_size:], axis=0, name='test_set')



pixel_count = img_height * img_width



# Reshape to 3D Tensor

X_train = np.array(X_train).reshape(train_size, pixel_count)

X_test = np.array(X_test).reshape(test_size, pixel_count)



# Normalise pixel values

X_train /= 255

X_test /= 255



# Training labels

Y_train_ = np.array(labels).reshape(len(labels))



# Check shape of each tensor

print(X_train.shape, Y_train_.shape)

print(X_test.shape)
# Show number of unique labels

classes = np.unique(Y_train_)

print(classes)

n_classes = len(np.unique(Y_train_))



# Create classes from unique labels

Y_train = to_categorical(Y_train_, n_classes)

print(Y_train.shape)
model = Sequential()



model.add(Dense(512, activation='relu', 

                 input_shape=(pixel_count,)))



model.add(Dense(512, activation='relu'))



model.add(Dense(512, activation='relu'))



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(n_classes, activation = 'softmax'))
model.compile(optimizer = 'adam', 

             loss = 'categorical_crossentropy', 

             metrics = ['accuracy'] )
history = model.fit(X_train, Y_train, batch_size=32, epochs=25, verbose=2)
predictions = model.predict_classes(X_test)

print(predictions)
df = pd.DataFrame({'Image name': names[-test_size:], 'Rotated angle (deg)': classes[predictions]})

df.head(10)
fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')



plt.subplot(2,1,2)

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')



plt.tight_layout()