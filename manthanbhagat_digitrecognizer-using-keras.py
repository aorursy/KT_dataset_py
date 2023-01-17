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
import numpy as np 
%matplotlib inline
from matplotlib import pyplot as plt # to view digits images from array object
from keras import  backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import  Sequential
from keras.layers.core import  Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D , MaxPooling2D ,Activation
from keras.optimizers import Adam ,RMSprop
train_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_set = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

img_col = 28
img_row = 28
train_set.head()
test_set.head()
X_train_df = train_set.drop(['label'],axis = 1)
y_train_df = train_set['label']
X_test_df = test_set

X_tr = np.asarray(X_train_df)/255
y_tr = np.asarray(y_train_df)
X_te = np.asarray(X_test_df)/255
print(type(X_tr))
print(X_tr.shape)
print(X_te.shape)

X_trainplot = X_tr.reshape(42000 , img_col , img_row)
X_testplot = X_te.reshape(28000, img_col,img_row)
print(X_trainplot.shape)
print(X_testplot.shape)
rand_img = np.random.randint(42000,size = 3)
fig = plt.figure()

for i,idx in enumerate(rand_img,1):
    arr1 = X_trainplot[idx]
    ax1 = fig.add_subplot(2,3,i)
    ax1.imshow(arr1 , cmap = 'gray')
    ax1.set_title(y_tr[idx])
    
rand_img = np.random.randint(28000,size = 3)
for i,idx in enumerate(rand_img,4):
    arr1 = X_testplot[idx]
    ax1 = fig.add_subplot(2,3,i)
    ax1.imshow(arr1 , cmap = 'gray')
X_train_f = X_tr.reshape(42000,img_col,img_row,1)
X_test_f= X_te.reshape(28000,img_col,img_row,1)
y_train_f = to_categorical(y_tr)
y_train_f.shape[1]
model = Sequential()

model.add(Conv2D(64,(2,2),padding = 'same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(2,2),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10 , activation = 'softmax'))
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 150
batch_size = 64

X_train, X_val, y_train, y_val = train_test_split(X_train_f, y_train_f, test_size=0.2 , random_state = 25)
image_gen = ImageDataGenerator(rotation_range = 5 ,width_shift_range = 0.15,height_shift_range = 0.15 ,shear_range = 0.3,zoom_range = 0.08)

train_batches = image_gen.flow(X_train,y_train,batch_size = batch_size)
val_batches =image_gen.flow(X_val,y_val,batch_size = batch_size)
steps_per_epoch = train_batches.n//train_batches.batch_size
validation_steps = val_batches.n//val_batches.batch_size

history=model.fit_generator(generator=train_batches, steps_per_epoch = steps_per_epoch, epochs=epochs, 
                    validation_data=val_batches, validation_steps=validation_steps)
predictions = model.predict_classes(X_test_f, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("mysub2.csv", index=False, header=True)