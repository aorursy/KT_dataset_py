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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array



from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data_generator = ImageDataGenerator(rotation_range=45,width_shift_range=8,height_shift_range=4,zoom_range=(0.8,1.2),horizontal_flip=True,channel_shift_range=0.2)



img_size = (224, 224)

img_array_list = []

cls_list = []



for n in range(10):

    img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/' + str(n) + '/*.jpg')

    for i in img_list1:

        img = load_img(i, color_mode='grayscale', target_size=(img_size))

        img_array = img_to_array(img) / 255

        img_array_list.append(img_array)

        cls_list.append(n)

#        ex_img = image_data_generator.flow(img_array.reshape(1,224,224,1),batch_size=1,shuffle=False)

#        img_array_list.append(ex_img)

#        cls_list.append(n)



X_train = np.array(img_array_list)

y_train = np.array(cls_list)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data_generator = ImageDataGenerator(rotation_range=45,width_shift_range=8,height_shift_range=4,zoom_range=(0.8,1.2),horizontal_flip=True,channel_shift_range=0.2)
X_train.shape
ex_img = np.empty([7164,224,224,1])

ex_img = image_data_generator.flow(X_train[:7164],batch_size=7164,shuffle=False)[0]
for i in range(7164):

    img_array_list.append(ex_img[i])

    cls_list.append(cls_list[i])
X_train = np.array(img_array_list)

y_train = np.array(cls_list)
X_train.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense



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



model.add(Dense(units=10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train,10)
model.fit(X_train, y_train,epochs=12,batch_size=64)
img_array_list = []



img_list = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/test/*.jpg')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)
predict = np.argmax(model.predict(X_test), axis=1)



submit = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

submit['class'] = predict

submit.to_csv('submission.csv', index=False)
X_train.shape
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(20, 20))

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.1, wspace=-0.9)

for i in range(100):

    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])

    ax.imshow(ex_img[i].reshape((224, 224)), cmap='gray')