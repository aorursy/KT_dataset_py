#!/usr/bin/env python

# coding: utf-8



# In[17]:





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





# In[18]:





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,  BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler



#sns.set(style='white', context='notebook', palette='deep')



# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")





# In[19]:





Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)



Y_train.value_counts()





# In[20]:





X_train.isnull().any().describe()





# In[21]:





test.isnull().any().describe()





# In[22]:





# 0~255???0~1????????????

X_train = X_train / 255.0

test = test / 255.0



# 1??784???28??28?????????(1?????????2???????????????)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



#????????????one hot vectors??? (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)



random_seed = 2

# ????????????????????????

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

#?????????????????????????????????????????????





# In[23]:





g = plt.imshow(X_train[1][:,:,0])





# In[24]:





# CNN model

model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.1))



#model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.1))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))





# In[25]:





optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["acc"])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)





annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)





datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.1,  

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        )  

datagen.fit(X_train)



hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

                           steps_per_epoch=1000,

                           epochs=1000, 

                           verbose=2,  

                           validation_data=(X_train[:400,:], Y_train[:400,:]), #For speed

                           callbacks=[annealer])



final_loss, final_acc = model.evaluate(X_train, Y_train, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))







# predict results

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST_ver21.csv",index=False)