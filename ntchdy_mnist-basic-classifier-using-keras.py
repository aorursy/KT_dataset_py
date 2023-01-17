import numpy as np

import pandas as pd

import pickle

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
img_rows, img_cols = 28, 28 

num_classes = 10
def prep_data(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)

    

    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images,img_rows, img_cols, 1)

    return x_shaped_array/255, out_y
def prep_test(raw):

    return raw.values.reshape(-1,img_rows,img_cols,1)/255
train_data = pd.read_csv("../input/train.csv")

x,y = prep_data(train_data)
model = Sequential()



model.add(Conv2D(42,kernel_size=(3,3), strides = 2, activation='relu', input_shape=(img_rows, img_cols,1)))

model.add(Dropout(0.8))

model.add(Conv2D(42, kernel_size=(3,3), strides = 2, activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer = 'adam', metrics=['accuracy'])
model.summary()
model.fit(x,y, batch_size = 128, epochs = 3, validation_split=0.2)
model.save('../working/my_model.h5')
test_data = pd.read_csv('../input/test.csv')

x_test = prep_test(test_data)

x_test.shape
result = model.predict(x_test)
result = np.argmax(result, axis = 1)

result = pd.Series(result,name='Label')

result
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),result],axis=1)

submission.head()
submission.to_csv("../working/submit.csv",index=False)