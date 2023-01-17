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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, Activation,BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()
train.shape
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
test.shape
# creating X and y 
y_train=train['label']
train=train.drop(['label'],axis=1)
# reshaping the data to feed in the model

train=train.values.reshape(train.shape[0],28,28,1)
test=test.values.reshape(test.shape[0],28,28,1)
print('The shape of X_train is: ', train.shape)
print('The shape of X_test is: ', test.shape)

print('The shape of y_train is: ', y_train.shape)
# With data augmentation to prevent overfitting (accuracy 0.99286)

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


datagen.fit(train)
## train-test-split

X_train, X_test, y_train, y_test = train_test_split(train,y_train,random_state=100,test_size=0.1)
# Define the optimizer
optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
def model():
    model=Sequential()
    model.add(Convolution2D(32,(5,5),input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(32,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32,(5,5),input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(32,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    return model
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)
model=model()
## epochs=500 used for actual submission
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=20,validation_batch_size=20,
         callbacks=learning_rate_reduction)
prediction=np.argmax(model.predict(test),axis=1)
prediction
submission=pd.DataFrame({'ImageID':range(1,28001),'Label':prediction})
submission.to_csv('Submission.csv',index=False)
from matplotlib import pyplot as plt


num = 4
print("Image label is:| ", prediction[num])
plt.imshow(test[num][:,:,0])
