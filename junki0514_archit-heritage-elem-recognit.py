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

from keras.utils.np_utils import to_categorical



#add

from keras.preprocessing.image import ImageDataGenerator



img_size = (224, 224)

img_array_list = []

cls_list = []



#add

datagen = ImageDataGenerator(horizontal_flip = 0.3)





#Class0

img_list0 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/0/*.jpg')

for i in img_list0:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(0)

    

#Class1

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/1/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(1)

        

#Class2

img_list2 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/2/*.jpg')

for i in img_list2:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(2)

    

#Class3

img_list3 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/3/*.jpg')

for i in img_list3:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(3)

    

#Class4

img_list4 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/4/*.jpg')

for i in img_list4:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(4)

        

#Class5

img_list5 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/5/*.jpg')

for i in img_list5:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(5)

    

#Class6

img_list6 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/6/*.jpg')

for i in img_list6:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(6)

    

#Class7

img_list7 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/7/*.jpg')

for i in img_list7:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(7)



#Class8

img_list8 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/8/*.jpg')

for i in img_list8:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(8)

    

#Class9

img_list9 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/9/*.jpg')

for i in img_list9:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(9)

    

X_train = np.array(img_array_list)

y_train = to_categorical(np.array(cls_list))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation



model = Sequential()



model.add(Conv2D(24, 3, padding='same', input_shape=(img_size[0], img_size[1], 1)))

model.add(Activation('relu'))



model.add(Conv2D(48, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))





model.add(Conv2D(96, 3, padding='same'))

model.add(Activation('relu'))



model.add(Conv2D(96, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

    

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

    

model.add(Dense(units=10, activation='softmax'))

    

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=22)
model.fit(X_train,y_train)
import glob

from keras.preprocessing.image import load_img, img_to_array



img_array_list = []



img_list = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/test/*.jpg')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)
X_test
predict = model.predict(X_test)
for i in range(3071):

    predict[i] = predict[i].argmax()
predict
predict = predict.astype('int32')
submit = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

submit['class'] = predict

submit.to_csv('submission.csv', index=False)
submit['class']