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
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from keras import regularizers
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_train = train.drop('label',axis=1)
Y_train = train['label']
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0
test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train,10)
random_seed = 42
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),strides=1,padding='Same',input_shape=[28,28,1],kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(filters=32,kernel_size=(5,5),strides=1,padding='Same',kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(5,5),strides=1,padding='Same',kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(filters=64,kernel_size=(5,5),strides=1,padding='Same',kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1, factor=0.5,min_lr=0.00001)
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=[0.9,1.1],
                            horizontal_flip=False,
                            vertical_flip=False)
datagen.fit(X_train)
epochs = 20
batch_size = 32
model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,steps_per_epoch=(X_train.shape[0]/batch_size),
                   validation_data=(X_cv,Y_cv),callbacks=[learning_rate_reduction])
Y_pred = model.predict(X_cv)
Y_pred_digit = np.argmax(Y_pred,axis=1)
Y_true = np.argmax(Y_cv,axis=1)
confusion_matrix(Y_true,Y_pred_digit)
results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("digit_recognizer_data_augmentation.csv",index=False)
