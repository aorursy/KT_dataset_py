# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        '''print(os.path.join(dirname, filename))'''



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras.utils.np_utils import to_categorical



img_size = (128, 128)

img_array_list = []

cls_list = []



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/0/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(0)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/1/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(1)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/2/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(2)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/3/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(3)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/4/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(4)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/5/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(5)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/6/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(6)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/7/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(7)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/8/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(8)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/9/*.jpg')

for i in img_list1:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(9)

    

X_train = np.array(img_array_list)

y_train = to_categorical(np.array(cls_list))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from tensorflow.keras.metrics import AUC



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))

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



model.add(Dense(units=10, activation='softmax'))



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=15)
from keras.preprocessing.image import load_img, img_to_array



img_array_list_test = []



img_list = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/test/*.jpg')

img_list.sort()

for i in img_list:

    img = load_img(i, target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list_test.append(img_array)



X_test = np.array(img_array_list_test)
p = model.predict(X_test)

predict = np.argmax(p, axis=1)



submit = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

submit['class'] = predict

submit.to_csv('submission.csv', index=False)