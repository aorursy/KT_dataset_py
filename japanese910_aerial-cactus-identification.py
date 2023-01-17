# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



"""import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))"""



# Any results you write to the current directory are saved as output.


import os

import cv2

import numpy as np

import pandas as pd

train_label = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")

submission = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv")



list_0 = []

list_1 = []



for i in train_label["has_cactus"]:

    if i == 0:

        list_0.append(i)

    else:

        list_1.append(i)

        

train_images = []

train_labels = []

test_images = []



for i in train_label["has_cactus"]:

    train_labels.append(i)



train_file = os.listdir("/kaggle/input/aerial-cactus-identification/train/train/")

train_file.sort()

test_file = os.listdir("/kaggle/input/aerial-cactus-identification/test/test/")

test_file.sort()



for image_name in train_file:

    img = cv2.imread("/kaggle/input/aerial-cactus-identification/train/train/"+image_name)

    train_images.append(img)



for image_name in test_file:

    img = cv2.imread("/kaggle/input/aerial-cactus-identification/test/test/"+image_name)

    test_images.append(img)

    

train_images = np.array(train_images)/255

train_labels = np.array(train_labels)

test_images = np.array(test_images)/255



train_images.shape, train_labels.shape, test_images.shape


from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Lambda, LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, ModelCheckpoint



model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(32, 32, 3)))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64,  (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

          

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))

          

model.add(Conv2D(128, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128,  (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))       

          

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))  

          

model.add(Conv2D(256, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

          

model.add(Conv2D(128, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

          

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))

          

model.add(Flatten())        

model.add(Dense(256,activation='relu',name='dense1'))

model.add(LeakyReLU(alpha=0.1))          

model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))
initial_learningrate=1e-3



def lr_decay(epoch):

    return initial_learningrate * 0.99 ** epoch



checkpoint = ModelCheckpoint(filepath='/kaggle/working/best_model.h5',monitor='val_loss', verbose=1, save_best_only=True)

    

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=initial_learningrate), metrics=["acc"])

model.fit(train_images,train_labels,epochs=100,batch_size=128,callbacks=[LearningRateScheduler(lr_decay,verbose=1),checkpoint],validation_split=0.2,verbose=1)
import datetime

from keras.models import load_model



best_model = load_model("/kaggle/working/best_model.h5")

pred = best_model.predict(test_images,verbose=1)



results = []

for i in range(len(pred)):

    if pred[i] >= 0.5:

        results.append(1)

    else:

        results.append(0)





submissions=pd.DataFrame({"id": submission["id"], "has_cactus": results})

submissions.to_csv("submission.csv", index=False)