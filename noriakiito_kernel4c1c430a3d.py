# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array



img_size = (224, 224)

img_array_list = []

cls_list = []





train_dir = '/kaggle/input/1056lab-defect-detection/train/Class'

for i in range(6):

    x = str(i + 1)

    img_list1 = glob.glob(train_dir + x + '/*.png')

    for i in img_list1:

        img = load_img(i, color_mode='grayscale', target_size=(img_size))

        img_array = img_to_array(img) / 255

        img_array_list.append(img_array)

        cls_list.append(0)



    img_list1 = glob.glob(train_dir + x + '_def/*.png')

    for i in img_list1:

        img = load_img(i, color_mode='grayscale', target_size=(img_size))

        img_array = img_to_array(img) / 255

        img_array_list.append(img_array)

        cls_list.append(1)



X_train = np.array(img_array_list)

y_train = np.array(cls_list)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import tensorflow.keras.optimizers as opt



model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(MaxPooling2D(pool_size=(8, 8)))

model.add(Dropout(rate=0.5))



model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Dropout(rate=0.5))

model.add(Flatten())

model.add(Dense(units=1024, activation='relu'))



model.add(Dense(units=1, activation='sigmoid'))

Nadam = opt.Nadam(lr=8e-4, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(optimizer=Nadam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
from keras.utils.np_utils import to_categorical

model.fit(X_train, y_train, epochs=250, batch_size=64)
import glob

from keras.preprocessing.image import load_img, img_to_array



img_array_list = []



img_list = glob.glob('/kaggle/input/1056lab-defect-detection/test/*.png')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)
predict = model.predict(X_test)[:, 0]



submit = pd.read_csv('/kaggle/input/1056lab-defect-detection/sampleSubmission.csv')

submit['defect'] = predict

submit.to_csv('submission8.csv', index=False)