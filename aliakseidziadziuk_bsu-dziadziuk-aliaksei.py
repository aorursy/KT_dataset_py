import tensorflow

import pandas as pd

import numpy as np

import keras

import keras.layers as layers

from keras.models import Sequential

from keras.utils.np_utils import to_categorical

from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from keras.callbacks import CSVLogger, ModelCheckpoint

from skimage.io import imshow, imshow_collection
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train.head(3)



X = train.iloc[:,1:].values

y = train.iloc[:,0].values

val = test.iloc[:,1:].values



y = keras.utils.to_categorical(y)



# X = np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')

X_flattened = X.reshape(X.shape[0],28,28,1)

X_flattened = np.pad(X_flattened, ((0,0),(2,2),(2,2),(0,0)), 'constant')

val_flattened = val.reshape(val.shape[0],28,28,1)

val_flattened = np.pad(val_flattened, ((0,0),(2,2),(2,2),(0,0)), 'constant')



X_rescaled = X_flattened/255.

val_rescaled = val_flattened/255.



x_train,x_test,y_train,y_test = train_test_split(X_rescaled,y,train_size=0.85,random_state=10)
model = keras.Sequential() 

model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='tanh', input_shape=(32,32,1)))

model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='tanh', input_shape=(32,32,1)))

model.add(layers.AveragePooling2D())

model.add(layers.BatchNormalization())



model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))

model.add(layers.AveragePooling2D())

model.add(layers.BatchNormalization())



model.add(layers.Flatten())



model.add(layers.Dense(units=120, activation='tanh'))



model.add(layers.Dense(units=84, activation='tanh'))



model.add(layers.Dense(units=10, activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train ,y_train, epochs = 30, batch_size=16, validation_data=[x_test,y_test], callbacks=[ModelCheckpoint('./model.h5', save_best_only=True)])
from keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)
model.load_weights('./model.h5');

predictions = model.predict_classes(val_rescaled)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = predictions
submission.to_csv("submission.csv",index=False)