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
import cv2

import matplotlib.pyplot as plt
data=os.listdir("../input/train/train") #path for image data
data
categories=[] #empty list

for x in data:

    category=x.split('.')[0]

    if(category=='dog'):

        categories.append(1)

    else:

        categories.append(0)
df=pd.DataFrame()
df["filename"]=data

df["category"]=categories
df.head(10)
df['category'].value_counts().plot.bar()

df.head(10)
import random

sample=random.choice(data) #picking random sample from data list

img=cv2.imread("../input/train/train/"+sample)

plt.imshow(img,cmap="gray")
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3 # RGB color
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64,(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128,(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
from sklearn.model_selection import train_test_split
train_df,validate_df=train_test_split(df,test_size=0.2,random_state=42)

train_df = train_df.reset_index(drop='True')

validate_df = validate_df.reset_index(drop='True')
train_df
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
train_df.shape
validate_df.shape
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical
train_df
train_df.info()
train_datagen=ImageDataGenerator(

                    rotation_range=15,

                    rescale=1./255,

                    shear_range=0.1,

                    zoom_range=0.2, # zoom range (1-0.2 to 1+0.2)

                    horizontal_flip=True,

                    width_shift_range=0.1,

                    height_shift_range=0.1

                 )

train_generator=train_datagen.flow_from_dataframe(

                    dataframe=train_df, 

                    directory="../input/train/train/", 

                    x_col="filename",

                    y_col="category",

                    target_size=IMAGE_SIZE,

                    class_mode='other',

                    batch_size=15

                )
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

        validate_df, 

        "../input/train/train/", 

        x_col='filename',

        y_col='category',

        target_size=IMAGE_SIZE,

        class_mode='other',

        batch_size=15

    )
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/train/train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='other'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        #print(Y_batch[0])

        break
batch_size=15
epochs=3

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=train_df.shape[0]//batch_size,

    steps_per_epoch=train_df.shape[0]//batch_size,

    callbacks=callbacks

)
test_file=os.listdir("../input/test1/test1")

test_df=pd.DataFrame()

test_df["filename"]=test_file
test_df
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/test1/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(test_df.shape[0]/batch_size))
predict
threshold=0.5

test_df["category"]=predict

test_df['category'] = np.where(test_df['category'] > threshold, 1,0)
test_df
sample=random.choice(test_file)

print(sample)

img=cv2.imread("../input/test1/test1/"+sample)

plt.imshow(img)
test_sample = test_df.head(10)

test_sample.head()



plt.figure(figsize=(12, 24))

for index, row in test_sample.iterrows():

    filename = row['filename']

    category = row['category']

    img = cv2.imread("../input/test1/test1/"+filename)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')')


