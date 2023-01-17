# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

# prepare the bg

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')





train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")







OX_train=train.drop('label',axis=1)

Oy_train=train.label



X_train, X_val, y_train, y_val = train_test_split(OX_train, Oy_train, test_size=0.3, random_state=0)

print(X_train.shape)

print(X_val.shape)

print(y_train.shape)

print(y_val.shape)
#normalization

X_train = X_train / 255.0

X_val = X_val / 255.0

test = test / 255.0

#reshape

X_train = X_train.values.reshape(-1,28,28,1)

X_val = X_val.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)





# Some examples

g = plt.imshow(X_train[8][:,:,0])
#categorical

y_train = to_categorical(y_train, num_classes = 10)

y_val = to_categorical(y_val,num_classes = 10)
#padding

import numpy as np

 

# Pad images with 0s

# X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    

# print("Updated Image Shape: {}".format(X_train[0].shape))
#augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
#modeling

from keras.layers import BatchNormalization

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
from keras.optimizers import Adam

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
#model complile

model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs=5 #change this to 30 if you need to get better score

batch_size=64
#fit the model

# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data =(X_val,y_val) ,

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
y_pre_test1=model.predict(test)

y_pre_test=np.argmax(y_pre_test1,axis=1)

csv_to_submit = pd.DataFrame(y_pre_test,columns=['Label'])

#csv_to_submit.to_csv('result.csv',index = True)