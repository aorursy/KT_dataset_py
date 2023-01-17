import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

print(os.listdir("../input"))



from zipfile import ZipFile



import numpy as np
import pandas as pd

test = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
import zipfile

zipref = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip')



zipref.extractall()
zipref = zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip')



zipref.extractall()
import os, sys



len(os.listdir('train/'))

len(os.listdir('test/'))
train.head()
test.head()
train.has_cactus=train.has_cactus.astype(str)
print('out dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
train['has_cactus'].value_counts()
Cactus_Proportion = print(13136/17500)

# In 75% of photos there is a cactus 
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
Image(os.path.join("train",train.iloc[0,0]),width=250,height=250)
train_slice = (train[:15001]) 
train_val_slice = (train[15000:])
train_generator=datagen.flow_from_dataframe(dataframe=train,directory=train,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator=datagen.flow_from_dataframe(dataframe=train,directory=train,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(),metrics=['acc'])
epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()