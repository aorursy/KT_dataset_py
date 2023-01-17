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
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import random
import gc
train_dir = "../input/train"
test_dir = "../input/test"


train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

test_img = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_imgs)

del train_dogs
del train_cats
gc.collect()
train_imgs
import matplotlib.image as mpimg
for ima in train_imgs[6:10]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()
nrows = 150
ncolumns =150
channels = 3

def read_and_process_image(list_of_images):
    
    x = [] # images
    y = [] # labels
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
            
            
    return x, y
        

x, y = read_and_process_image(train_imgs)
x[0]
y
plt.figure(figsize = (20,10))
columns =5
for i in range(columns):
    plt.subplot(2, columns, i+1)
    plt.imshow(x[i])
import seaborn as sns
del train_imgs
gc.collect()

x = np.array(x)
y = np.array(y)

sns.countplot(y)
plt.title('Labels of cats and dogs')
print('shape of train image is: ', x.shape)
print('shape of label is: ', y.shape)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=2)

print('shape of train image is: ', X_train.shape)
print('shape of validation image is: ', X_val.shape)
print('shape of labels is: ', y_train.shape)
print('shape of label is: ', y_val.shape)
del x
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)

batch_size = 32
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

val_datgen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datgen.flow(X_val, y_val, batch_size=batch_size)
history = model.fit_generator(train_generator,
                             steps_per_epoch=ntrain // batch_size,
                             epochs=64,
                             validation_data=val_generator,
                             validation_steps=nval // batch_size)
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('training and validation accuracy')

plt.figure

plt.plot(epochs, loss, 'b', label = 'training loss')
plt.plot(epochs, val_loss, 'r', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()
X_test, y_test = read_and_process_image(test_img[:10])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale = 1./255)
i = 0
text_labels = []
plt.figure(figsize=(20,10))
for batch in test_datagen.flow(x, batch_size = 1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    
    plt.subplot(5/columns + 1, columns, i+1)
    plt.title("This is a " + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i +=1
    if i % 10 == 0:
        break
plt.show()
