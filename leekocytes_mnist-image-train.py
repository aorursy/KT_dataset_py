import numpy as np

import keras

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split





from keras import models

from keras import optimizers

from keras.layers import *

from keras.optimizers import *
train_data = pd.read_csv('../input/train.csv')

Y = np.asarray(train_data['label'])

X = train_data.drop('label',axis = 1)

X=np.asarray(X)

X.shape, Y.shape
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2 , random_state=42)


k_submit = pd.read_csv('../input/test.csv')

k_test=np.asarray(k_submit)

k_test.shape
def build_model():

    model = models.Sequential()

    model.add(Dense(32, 

                    activation='relu',

                    input_dim = 784))

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(32, activation = 'relu'))

    model.add(Dense(10, activation = 'sigmoid'))

    model.summary()

    Adam= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer = Adam, loss = 'categorical_crossentropy', metrics = ['acc'])

    return model

model = build_model()
y_train = keras.utils.to_categorical(y_train).astype('int32')

y_test = keras.utils.to_categorical(y_test).astype('int32')

y_train.shape
history = model.fit(x_train,y_train,

                   epochs = 50,

                   validation_split = 0.2,

                   batch_size = 128)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'g', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.figure()



plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'g', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.show()
model.evaluate(x_test,y_test)
model.metrics_names
k_test.shape
res = model.predict(k_test)
res.shape
res = np.argmax(res,axis = 1)
res.shape
res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)

submission.to_csv("submission.csv",index=False)

submission.head(10)