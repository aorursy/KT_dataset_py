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

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras.utils.np_utils import to_categorical



img_size = (224, 224)

img_array_list = []

cls_list = []



img_list1 = glob.glob('/kaggle/input/1056lab-defect-detection-extra/train/Class*/*.png')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(0)



img_list1 = glob.glob('/kaggle/input/1056lab-defect-detection-extra/train/Class*_def/*.png')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(1)



X_train = np.array(img_array_list)

y_train = to_categorical(np.array(cls_list))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from tensorflow.keras.metrics import AUC



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.25))



model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(rate=0.25))



model.add(Dense(units=2, activation='softmax'))



auc = AUC()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc])

model.summary()
model.fit(X_train, y_train, epochs=15, batch_size=None)
import glob

from keras.preprocessing.image import load_img, img_to_array



img_array_list = []



img_list = glob.glob('/kaggle/input/1056lab-defect-detection-extra/test/*.png')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)
predict = model.predict(X_test)[:, 1]



submit = pd.read_csv('/kaggle/input/1056lab-defect-detection-extra/sampleSubmission.csv')

submit['defect'] = predict

submit.to_csv('submission.csv', index=False)