# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install -U efficientnet
train_dir = '../input/train/train'

test_dir = '../input/test'

train_df = pd.read_csv('../input/train.csv')

train_df.head()
import cv2

cactus = []

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][0]))

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][1]))

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][2]))

#now reading no cactus images

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][6]))

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][7]))

cactus.append(cv2.imread(train_dir + '/' + train_df['id'][11]))





labels = ['cactus','cactus','cactus','no cactus','no cactus',' no cactus']



import matplotlib.pyplot as plt



plt.figure(figsize=[10,10])

for x in range(0,6):

    plt.subplot(3, 3,x+1)

    plt.imshow(cactus[x])

    plt.title(labels[x])

    x += 1

    

plt.show()
from keras import applications

from efficientnet import EfficientNetB3

from keras import callbacks

from keras.models import Sequential
train_df['has_cactus'] = train_df['has_cactus'].astype('str')
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rescale=1/255,

    validation_split=0.10,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(32,32),

    subset="training",

    batch_size=1024,

    shuffle=True,

    class_mode="binary"

)



val_generator = train_datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(32,32),

    subset="validation",

    batch_size=256,

    shuffle=True,

    class_mode="binary"

)
test_datagen = ImageDataGenerator(

    rescale=1/255

)



test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(32,32),

    batch_size=1,

    shuffle=False,

    class_mode=None

)
from keras.layers import Dense

from keras.optimizers import Adam



efficient_net = EfficientNetB3(

    weights='imagenet',

    input_shape=(32,32,3),

    include_top=False,

    pooling='max'

)



model = Sequential()

model.add(efficient_net)

model.add(Dense(units = 120, activation='relu'))

model.add(Dense(units = 120, activation = 'relu'))

model.add(Dense(units = 1, activation='sigmoid'))

model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(

    train_generator,

    epochs = 50,

    steps_per_epoch = 15,

    validation_data = val_generator,

    validation_steps = 7

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1)



plt.plot(epochs,acc,'bo',label = 'Training Accuracy')

plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs,loss,'bo',label = 'Training loss')

plt.plot(epochs,val_loss,'b',label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()





preds = model.predict_generator(

    test_generator,

    steps=len(test_generator.filenames)

)
image_ids = [name.split('/')[-1] for name in test_generator.filenames]

predictions = preds.flatten()

data = {'id': image_ids, 'has_cactus':predictions} 

submission = pd.DataFrame(data)

print(submission.head())
submission.to_csv("submission.csv", index=False)