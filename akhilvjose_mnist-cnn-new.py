# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
Y_train=train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
test.shape
#ALTERNATIVE METHOD:

# train = train.values
# np.random.shuffle(train)

# X = train[:, 1:].reshape(-1, 28, 28, 1) / 255.0
# Y = train[:, 0].astype(np.int32)

#test = test.values
#test = test.reshape[-1,28,28,1]
Y_train.value_counts() #pandas
Y_train = to_categorical(Y_train, num_classes = 10)
# keras.utils.to_categorical(y, num_classes=None, dtype='float32')
# https://keras.io/utils/
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
g = plt.imshow(X_train[0][:,:,0])
image=Input(shape=[28,28,1])

x= Conv2D(128, (3, 3), padding='same', activation='relu')(image)
x=MaxPooling2D()(x)

x= Conv2D(128, (3, 3), padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D(strides = (1,1))(x)

x= Conv2D(128, (3, 3), padding='same')(x)
x=BatchNormalization()(x)
x=Activation('relu')(x)
x=MaxPooling2D()(x)

x=Flatten()(x)

x= Dense(2000, activation = 'relu')(x)
x= Dropout(0.5, noise_shape=None, seed=None)(x)
x= Dense(1000, activation = 'relu')(x)
x= Dropout(0.3, noise_shape=None, seed=None)(x)
x= Dense(250, activation = 'relu')(x)
x= Dropout(0.3, noise_shape=None, seed=None)(x)
out = Dense(10,activation = 'softmax')(x)

model = Model(image,out )
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='Adam',
  metrics=['accuracy']
)
#keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, 
#samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, 
#zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
#height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, 
#channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, 
#rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
# Image_generator = ImageDataGenerator(rotation_range=0, 
#                                      width_shift_range=0.0,
#                                      height_shift_range=0.0,
#                                      zoom_range=0.0,
#                                      horizontal_flip=False, 
#                                      vertical_flip=False,
#                                      #validation_split=0.15
#                                     )
# Image_generator.fit(X_train)
#flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, 
#save_prefix='', save_format='png', subset=None)
#train_generator = 
#valid_generator = 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=False)
# r = model.fit_generator(Image_generator.flow(X_train, Y_train, batch_size=32),
#                        #validation_data = valid_generator,
#                         validation_data =(X_val,Y_val),
#                         steps_per_epoch=X_train.shape[0]// 32,
#                         epochs=15,
#                         callbacks = [reduce_lr],
#                         verbose=1,
#                         validation_steps=X_val.shape[0]//32,
#                          )
r = model.fit(X_train,Y_train,batch_size = 32,
              epochs= 10,
             verbose = 1,
              callbacks = [reduce_lr,early_stop],
             validation_data = (X_val,Y_val)
             )
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['acc'], color='b', label="Training accuracy")
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_result.csv",index=False)
!ls
