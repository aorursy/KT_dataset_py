# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from os import listdir

from PIL import Image as PImage

from matplotlib.pyplot import imshow

from PIL import Image

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm

%matplotlib inline



print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cactus_data=pd.read_csv('../input/train.csv')

cactus_data.head(5)
w=10

h=10

fig=plt.figure(figsize=(10, 10))

columns = 5

rows = 1

images_display=cactus_data.head(5)

image_name=list(np.array(images_display['id']))

img_label=list(np.array(images_display['has_cactus']))

for i in range(1, columns*rows +1):

    path="../input/train/train/"+image_name[i-1]

    img =Image.open(path, 'r')

    f=fig.add_subplot(rows, columns, i)

    f.title.set_text(img_label[i-1])

    plt.imshow(img)

plt.show()

train_image = []

for i in tqdm(range(cactus_data.shape[0])):

    img = image.load_img('../input/train/train/'+ cactus_data['id'][i], target_size=(32,32,1), grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

Xdata = np.array(train_image)
Xdata.shape
Ydata=np.array(cactus_data['has_cactus'].values)

Ydata= to_categorical(Ydata)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, random_state=42, test_size=0.2)
model.fit(X_train, Y_train, epochs=12, validation_data=(X_test, Y_test))
model.evaluate(X_test, Y_test, verbose=0)

un_test_img=[]

count=0

for i in os.listdir("../input/test/test/"):

    un_test_img.append(i)

    count+=1

un_test_image=[]

for i in tqdm(range(count)):

    img = image.load_img('../input/test/test/'+un_test_img[i], target_size=(32,32,3), grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    un_test_image.append(img)

un_test_img_array = np.array(un_test_image)
len(un_test_img)
output = model.predict_classes(un_test_img_array)
output
submission_save = pd.DataFrame()

submission_save['id'] = un_test_img

submission_save['has_cactus'] = output

submission_save.to_csv('submission.csv', header=True, index=False)

pd.read_csv('submission.csv')