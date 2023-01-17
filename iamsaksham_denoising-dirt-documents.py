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
import glob

import cv2

from matplotlib import pyplot as plt

%matplotlib inline
X_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train/*.png")

y_train_path = glob.glob("/kaggle/input/denoising-dirty-documents/train_cleaned/*.png")

X_test_path = glob.glob("/kaggle/input/denoising-dirty-documents/test/*.png")



input_shape = (258, 540, 1)
# img = cv2.imread(X_train_path[0], cv2.IMREAD_GRAYSCALE)

# plt.imshow(img)

# plt.show()
def load_images(path):

    image_list = []

    for pth in path:

        img = cv2.imread(pth, 0) # read grayscale image

        img = cv2.resize(img, (input_shape[1], input_shape[0]))

        img = img / 255.

        img = np.expand_dims(img, axis=-1)

        image_list.append(img)

    return image_list
X_train_all = load_images(X_train_path)

y_train_all = load_images(y_train_path)

X_test = load_images(X_test_path)



# convert list of images to np array

X_train_all = np.array(X_train_all)

y_train_all = np.array(y_train_all)

X_test = np.array(X_test)



print(X_train_all.shape)

print(y_train_all.shape)

print(X_test.shape)
# print(X_train_all[0].shape)

# plt.imshow(X_train_all[44])

# plt.show()
# train val split



from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, random_state=0)



print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow.keras import layers



tf.keras.backend.clear_session()  # For easy reset of notebook state.
model = tf.keras.Sequential()



# add convolutional layer to the model with relu activation

# 32 convolution filters used each of size 3x3

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))

model.add(layers.MaxPooling2D(2, padding='same'))

model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(layers.UpSampling2D((2,2)))

model.add(layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'))





model.summary()
model.compile(loss='mean_squared_error',

              optimizer='adam',

              metrics=['accuracy'])
num_epochs = 100

# batch size -Total number of training examples present in a single batch.

batch_size = 8

history = model.fit(X_train, y_train, epochs=num_epochs,

                    batch_size=batch_size, 

                    verbose=1,

                    validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0]) #Test loss: 0.0296396646054

print('Test accuracy:', score[1]) #Test accuracy: 0.9904

final_predictions = model.predict(X_test)
preds_0 = final_predictions[10] * 255.0

preds_0 = preds_0.reshape(258, 540)

x_test_0 = X_test[10] * 255.0

x_test_0 = x_test_0.reshape(258, 540)

plt.imshow(x_test_0, cmap='gray')
plt.imshow(preds_0, cmap='gray')
final_predictions = final_predictions.reshape(-1, 258, 540)



ids = []

vals = []

for i, f in enumerate(X_test_path):

    file = os.path.basename(f)

    imgid = int(file[:-4])

    test_img = cv2.imread(f, 0)

    img_shape = test_img.shape

    print('processing: {}'.format(imgid))

    print(img_shape)

    preds_reshaped = cv2.resize(final_predictions[i], (img_shape[1], img_shape[0]))

    for r in range(img_shape[0]):

        for c in range(img_shape[1]):

            ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))

            vals.append(preds_reshaped[r, c])



print('Writing to csv file')

pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)