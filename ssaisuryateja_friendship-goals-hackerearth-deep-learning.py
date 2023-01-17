# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from matplotlib import pyplot as plt

import os

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
imgloc=[]

label=[]

for dirname, _, filenames in os.walk('/kaggle/input/friendshipgoals/data/train/Adults'):

    for filename in filenames:

        imgloc.append((os.path.join(dirname, filename)))

        label.append(0)

for dirname, _, filenames in os.walk('/kaggle/input/friendshipgoals/data/train/Teenagers'):

    for filename in filenames:

        imgloc.append((os.path.join(dirname, filename)))

        label.append(1)

for dirname, _, filenames in os.walk('/kaggle/input/friendshipgoals/data/train/Toddler'):

    for filename in filenames:

        imgloc.append((os.path.join(dirname, filename)))

        label.append(2)
img=[]

for i in range(0, len(imgloc)):

    img1 = cv2.imread(imgloc[i],1)

    img2 = np.array(img1)

    img2 = cv2.resize(img2,(128,128))

    img.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.figure()

f, axarr = plt.subplots(5,5) 



for i in range(0,5):

    for j in range(0,5):

        axarr[i][j].imshow(img[(5*i)+j])
df = pd.read_csv('/kaggle/input/friendshipgoals/data/Test.csv')

id_test = df['Filename']
imgloc2=[]

for dirname, _, filenames in os.walk('/kaggle/input/friendshipgoals/data/test'):

    for filename in filenames:

        imgloc2.append((os.path.join(dirname, filename)))
imgx=[]

for i in range(0, len(imgloc2)):

    img1 = cv2.imread(imgloc2[i],1)

    img2 = np.array(img1)

    img2 = cv2.resize(img2,(128,128))

    imgx.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.figure()

f, axarr = plt.subplots(5,5) 



for i in range(0,5):

    for j in range(0,5):

        axarr[i][j].imshow(imgx[(5*i)+j])
np.array(img).shape
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.models import Sequential

model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),activation='tanh',input_shape=img[0].shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dense(10, activation='softmax'))
import keras

model.compile(loss=keras.losses.sparse_categorical_crossentropy,

              optimizer=keras.optimizers.SGD(lr=0.01),

              metrics=['accuracy'])
history = model.fit(np.array(img), np.array(label), batch_size=10, epochs=40, verbose=1)
pre=[]

pred = model.predict(np.array(np.array(imgx)))

for i in range(0,len(imgx)):

    p=pred[i][0]

    tmp=0

    for j in range(0,3):

        if pred[i][j]>p:

            p=pred[i][j]

            tmp=j

    pre.append(tmp)
final=[]

for i in range(len(pre)):

    if(pre[i]==0): final.append('Adults')

    elif(pre[i]==1): final.append('Teenagers')

    else: final.append('Toddler')
df3 = pd.DataFrame()

df3['Filename'] = id_test

df3['Category'] = final

df3.to_csv("./file.csv", sep=',',index=True)
plt.plot(history.history['loss'])

plt.plot(history.history['accuracy'])

plt.title('model loss')

plt.ylabel('loss/accuracy')

plt.xlabel('epoch')

plt.legend(['loss', 'accuracy'], loc='upper left')

plt.show()