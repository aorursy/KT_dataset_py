import numpy as np 

import pandas as pd 

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D,BatchNormalization

img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



train_size = 60000

train_file = "../input/fashion-mnist_train.csv"

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(x, y,

          batch_size=256,

          epochs=5,

          validation_split = 0.2)
test_file = "../input/fashion-mnist_test.csv"

test_data = pd.read_csv(test_file)

num_test = test_data.shape[0]

x_as_array_test = test_data.values[:,1:]

x_shaped_array_test = x_as_array_test.reshape(num_test, img_rows, img_cols, 1)

out_x_test = x_shaped_array_test / 255
Y_test = model.predict(out_x_test)

pd.DataFrame(Y_test).to_csv("sub.csv")

test=np.argmax(Y_test,axis=1)

image_id=[]

for i in range (len(Y_test)):

    image_id.append(i+1)

answer=pd.DataFrame({'ImageId': image_id,'Label':test})

answer.to_csv('res.csv',index=False)