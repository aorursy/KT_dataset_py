# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)











# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import glob

import cv2

from pathlib import Path

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D



data_dir = Path('../input/intel-image-classification')



train_dir = data_dir / 'seg_train' / 'seg_train'

test_dir  = data_dir / 'seg_test' / 'seg_test'

pred_dir  = data_dir / 'seg_pred' / 'seg_pred'
# Creating empty lists

X_train = []

Y_train = []

X_test  = []

Y_test  = []

X_pred  = []
# Getting files and appending to their respective lists

buildings_dir = train_dir / 'buildings'

buildings     = buildings_dir.glob('*.jpg')



forest_dir    = train_dir / 'forest'

forest        = forest_dir.glob('*.jpg')



glacier_dir   = train_dir / 'buildings'

glacier       = glacier_dir.glob('*.jpg')



mountain_dir  = train_dir / 'mountain'

mountain      = mountain_dir.glob('*.jpg')



sea_dir       = train_dir / 'sea'

sea           = sea_dir.glob('*.jpg')



street_dir    = train_dir / 'street'

street        = street_dir.glob('*.jpg')


height = 100 

width  = 100

for a in buildings:

    image = cv2.imread(str(a))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(0, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

for b in forest:

    image = cv2.imread(str(b))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(1, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

for c in glacier:

    image = cv2.imread(str(c))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(2, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

for d in mountain:

    image = cv2.imread(str(d))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(3, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

for e in sea:

    image = cv2.imread(str(e))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(4, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

for f in street:

    image = cv2.imread(str(f))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(5, num_classes=6)

    X_train.append((img))

    Y_train.append((label))

    

X_train = np.array(X_train)  

Y_train = np.array(Y_train)
X_train.shape ,Y_train.shape
buildings_dir = test_dir / 'buildings'

buildings     = buildings_dir.glob('*.jpg')



forest_dir    = test_dir / 'forest'

forest        = forest_dir.glob('*.jpg')



glacier_dir   = test_dir / 'buildings'

glacier       = glacier_dir.glob('*.jpg')



mountain_dir  = test_dir / 'mountain'

mountain      = mountain_dir.glob('*.jpg')



sea_dir       = test_dir / 'sea'

sea           = sea_dir.glob('*.jpg')



street_dir    = test_dir / 'street'

street        = street_dir.glob('*.jpg')
for a in buildings:

    image = cv2.imread(str(a))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(0, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

for b in forest:

    image = cv2.imread(str(b))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(1, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

for c in glacier:

    image = cv2.imread(str(c))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(2, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

for d in mountain:

    image = cv2.imread(str(d))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(3, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

for e in sea:

    image = cv2.imread(str(e))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(4, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

for f in street:

    image = cv2.imread(str(f))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    label = to_categorical(5, num_classes=6)

    X_test.append((img))

    Y_test.append((label))

    

X_test = np.array(X_test)  

Y_test = np.array(Y_test)
X_test.shape ,Y_test.shape
pred_all = pred_dir.glob('*.jpg')

for p in pred_all:

    image = cv2.imread(str(p))

    resizeimage = cv2.resize(image, (height,width))

    img = resizeimage.astype(np.float32)/255.

    X_pred.append((img))

    

X_pred = np.array(X_pred)



X_pred.shape
# introducing inception block



#conv_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)



#conv_3x3 = Conv2D(96, (1, 1), padding='same', activation='relu')(x)

#conv_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_3x3)



#conv_5x5 = Conv2D(16, (1, 1), padding='same', activation='relu')(x)

#conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(conv_5x5)



#pool_proj = MaxPooling2D((2,2), strides = (1,1),padding='same')(x)

#pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu')(pool_proj)



#concat = Concatenate(axis=3)

#x = concat([conv_1x1, conv_3x3 , conv_5x5 , pool_proj])
model=Sequential()

model.add(Conv2D(32, (3,3),padding="same",activation="relu",input_shape=(100,100,3)))

model.add(MaxPooling2D(2,2))



model.add(SeparableConv2D(64 , (3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(2,2))



model.add(SeparableConv2D(128,(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(SeparableConv2D(128,(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.8))

model.add(Dense(6,activation="softmax"))

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



# training the model



model.fit(X_train ,Y_train , batch_size=500 ,epochs=120)
model.evaluate(X_test,Y_test)