# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train.head()
X_train=train.drop("label", axis=1)

y_train=train["label"]

X_test=test
import seaborn as sns

g=sns.countplot(y_train)

print(y_train.value_counts())
#normalizing data

#greyscale normalization

X_train/=255.0

X_test/=255.0

#reshaping

X_train=X_train.values.reshape(-1,28,28,1)

X_test=X_test.values.reshape(-1,28,28,1)



from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train, num_classes=10)
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test= train_test_split(X_train,y_train,test_size=0.1,random_state=0)
plt.imshow(X_train[2][:,:,0])
from keras.models import Sequential

from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras.layers.convolutional import Conv2D



#initialising CNN

classifier=Sequential()



#1.convolution

classifier.add(Conv2D(64,3,3, input_shape=(28,28,1), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(64,3,3, activation='relu'))

classifier.add(BatchNormalization())

#2.pooling

classifier.add(Conv2D(64,3,3, activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.4))



#second convolution layer

classifier.add(Conv2D(64,3,3, activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(64,3,3, activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(64,3,3, activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Dropout(0.4))

#3.flattenting

classifier.add(Flatten())



#4.full connection

classifier.add(Dense(output_dim=256,activation='relu'))

#output layer

classifier.add(Dropout(0.4))

classifier.add(Dense(output_dim=512,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(Dense(output_dim=1024,activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim=10, activation='softmax'))
from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',

                   metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
from keras.callbacks import LearningRateScheduler

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
classifier.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=40, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(x_test[:400,:], y_test[:400,:]), #For speed

                           callbacks=[annealer])
result=classifier.predict(X_test)

result=pd.Series(np.argmax(result, axis=1), name='Label')

result
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)