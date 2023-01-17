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
import seaborn as sns
%matplotlib inline 
#data
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
#create duplicates
train_data = train.copy()
test_data = test.copy()
train_data.shape,test_data.shape
train_data.head()
target = train_data['label']
train_data.drop('label',axis=1,inplace=True)
target.value_counts()
sns.countplot(target);
#import keras
from keras import models
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization
model = models.Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()
X = train_data.to_numpy()
y = target.to_numpy()
test_data = test_data.to_numpy()
X = X.astype('float32')
test_data = test_data.astype('float32')
y = y.astype('float32')
#normalizing
X = X/255
test_data = test_data/255
X = X.reshape(-1,28,28,1)
test_data = test_data.reshape(-1,28,28,1)
from keras.utils import to_categorical
y = to_categorical(y)
model.compile(optimizer='Adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
generator.fit(X)
from keras.callbacks import ReduceLROnPlateau
learning_rate_decay = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.fit_generator(generator.flow(X,y, batch_size=32),
                    epochs = 30,verbose = 2, steps_per_epoch=X.shape[0] // 32,callbacks=[learning_rate_decay])
ypreds = model.predict(test_data)
preds = np.argmax(ypreds,axis = 1)
submit=pd.DataFrame()
submit['ImageId']=range(1,28001)
submit['Label']=preds
submit.to_csv('submit.csv',index=False)
