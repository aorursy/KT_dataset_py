# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cell_images/cell_images/"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.optimizers import SGD
# layerの作成

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(rate=0.5))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the CNN



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
#　画像の入力、学習、テストデータに落とす

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   validation_split=0.3,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

train_set = train_datagen.flow_from_directory('../input/cell_images/cell_images/',

                                                 target_size = (64, 64),

                                                 batch_size =32,

                                                 class_mode = 'binary',

                                                 subset='training')

test_set = train_datagen.flow_from_directory('../input/cell_images/cell_images/',

                                                 target_size = (64, 64),

                                                 batch_size =32,

                                                 class_mode = 'binary',

                                                 subset='validation')
classifier.fit_generator(train_set,

                           steps_per_epoch=345,

                           epochs=10,

                           validation_data=test_set,

                           validation_steps=86)
hist = classifier.history.history
hist.keys()

val_acc=hist['val_acc']

val_loss=hist['val_loss']

acc=hist['acc']

loss=hist['loss']
import matplotlib.pyplot as plt
%matplotlib inline
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_Accuracy')

plt.plot(epochs,val_acc,'b',label='Validation_Accuracy')

plt.figure()

plt.show()
hist['acc']
hist['val_acc']
hist['val_loss']
plt.plot(epochs,loss,'bo',label='Training_Loss')

plt.plot(epochs,val_loss,'b',label='Validation_Loss')

plt.show()
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_Accuracy',color='r')

plt.plot(epochs,val_acc,'b',label='Validation_Accuracy',color='r')

plt.plot(epochs,loss,'bo',label='Training_Loss')

plt.plot(epochs,val_loss,'b',label='Validation_Loss')

plt.legend()

plt.show()