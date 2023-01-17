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
# Disable warning

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input/super-ai-image-classification/train/train/"))



import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
train_data = pd.read_csv("../input/super-ai-image-classification/train/train/train.csv")
train_data.info
train_data.shape[0]
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm
# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False

# https://morioh.com/p/fa76d93bfca8

# https://www.kaggle.com/lavanyask/lego-minifigures-classify



train_image = []

for i in tqdm(range(train_data.shape[0])):

    img = image.load_img('../input/super-ai-image-classification/train/train/images/'+train_data['id'][i], target_size=(224,224,3))

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
y = train_data['category'].values

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#model = Sequential(name = "Nachod_model")

#model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3)))

#model.add(Conv2D(128, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))

#model.add(Flatten())

#model.add(Dense(128, activation='relu'))

#model.add(Dropout(0.5))

#model.add(Dense(2, activation='softmax'))



model = Sequential(name = "Nachod_model")

model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=2, activation="softmax"))





#from tensorflow.keras import layers

#from tensorflow.keras import models

#from tensorflow.keras.applications.vgg16 import VGG16

#base_model = VGG16(input_shape = (224,224,3), include_top = False, weights = 'imagenet')

#for layer in base_model.layers:

#    layer.trainable = False   

#x = layers.Flatten()(base_model.output)

#x = layers.Dense(512, activation = 'relu')(x)

#x = layers.Dropout(0.5)(x)

#x = layers.Dense(2, activation = 'softmax')(x)

#model = models.Model(inputs = base_model.input, outputs = x, name = "Nachod_model")
model.summary()
#model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.0001), metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'min', restore_best_weights = True)
checkpoint = ModelCheckpoint('nachod_model.hdf5', monitor = 'val_loss', verbose = 1, mode = 'min', save_best_only = True)
#history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))



history = model.fit(X_train, y_train, epochs = 50, verbose = 1, validation_data=(X_test, y_test), callbacks = [early_stopping, checkpoint])
print(history.history.keys())
import matplotlib.pyplot as plt

print(history.history.keys())



# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
import glob



val_dir = '../input/super-ai-image-classification/val/val/images/'

test_image = []



file_list = glob.glob("{}/*.jpg".format(val_dir))
for i in tqdm(range(len(file_list))):

    img = image.load_img(file_list[i], target_size=(224,224,3))

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)

test = np.array(test_image)

id_file = np.array(os.listdir(val_dir))
test
model = models.load_model('nachod_model.hdf5')



prediction = model.predict_classes(test)
prediction
id_file
prediction.shape
id_file.shape
sample = pd.read_csv('../input/super-ai-image-classification/val/val/val.csv')

df = pd.DataFrame({'id' : id_file, 'category' : prediction})

df.to_csv('sample.csv', header=True, index=False)