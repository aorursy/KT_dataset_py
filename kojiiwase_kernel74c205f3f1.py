

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import os

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential,Model

import tensorflow.keras as keras

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')
train.shape
test.shape
test
import matplotlib.pyplot as plt
test=test.drop('id',axis=1)



test
y=train.label.value_counts()

sns.barplot(y.index,y)
X_train=train.drop('label',axis=1)
y_train=train['label']
X_train=X_train/255

test=test/255
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
print(X_train.shape)

print(test.shape)
y_train=to_categorical(y_train)
y_train
X_train,
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,

random_state=42,test_size=0.3)

plt.imshow(X_train[0][:,:,0])
datagen=ImageDataGenerator(

    width_shift_range=0.4,

    height_shift_range=0.4,

    rotation_range=30

)
model=Sequential()



model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',

                 kernel_initializer='he_normal', input_shape=(28, 28, 1)))  # 28x28x1 -> 24x24x16

model.add(MaxPooling2D(pool_size=(2, 2)))  # 24x24x16 -> 12x12x16

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',

                 kernel_initializer='he_normal'))  # 12x12x16 -> 8x8x64

model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8x64 -> 4x4x64



model.add(Flatten())  # 4x4x64-> 1024

model.add(Dense(10, activation='softmax'))  # 1024 -> 10



model.compile(

    loss=keras.losses.categorical_crossentropy,

    optimizer='adam',

    metrics=['accuracy']

)
early_stopping=EarlyStopping(patience=5,verbose=1)

model.fit(X_train,y_train,batch_size=1000,verbose=2,epochs=10,validation_data=(X_test,y_test),callbacks=[early_stopping])

y_pre=model.predict(test)
y_pre=np.argmax(y_pre,axis=1)
y_pre
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sample_sub
sample_sub['label']=y_pre
sample_sub
sample_sub.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv')